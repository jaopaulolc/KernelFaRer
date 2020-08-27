//===- GEMMReplacer.cpp - Matrix-Multiply Replacer Pass ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass replaces matrix-multiply loops previously recognized by the matcher
// into llvm.matrix.multiply.* intrinsic calls. In cases that this
// kicks in, it can be a significant performance win.
//
//===----------------------------------------------------------------------===//
//
// TODO List:
// * Add command-line replacement options for GEMM
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/GEMMFaRer.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "gemm-replacer-pass"

using namespace llvm;
using namespace GEMMFaRer;

// Anonymous namespace containing rewriter functions.
namespace {

// Command line argument that determines which mode we are replacing in.
cl::opt<GEMMFaRer::ReplacementMode> ReplaceMode(
    "gemmfarer-replacement-mode",
    cl::desc("Available GeMM replacement methods."),
    cl::values(clEnumValN(GEMMFaRer::MatrixIntrinsics, "matrix-intrinsics",
                          "Replace using llvm.matrix.* intrinsics."),
               clEnumValN(GEMMFaRer::CBLAS, "cblas-interface",
                          "Replace using the CBLAS interface.")),
    cl::ValueRequired, cl::init(GEMMFaRer::UNKNOWN));

// A helper function that returns Matrix loaded in column-major order as a
// flat-vector.
Value *loadMatrixToFlatVector(MatrixBuilder &MBuilder, Type &EltTy,
                              Value &Matrix, Value &Rows, Value &Columns,
                              Value &Stride, bool IsColMajor,
                              const Align &Alignment) {
  Value *Vec;
  uint64_t RowsAsUInt64 = cast<ConstantInt>(Rows).getZExtValue();
  uint64_t ColsAsUInt64 = cast<ConstantInt>(Columns).getZExtValue();

  if (IsColMajor)
    Vec = MBuilder.CreateColumnMajorLoad(&EltTy, &Matrix, Alignment, &Stride,
                                         /*isVolatile*/ false, RowsAsUInt64,
                                         ColsAsUInt64);
  else
    Vec = MBuilder.CreateMatrixTranspose(
        MBuilder.CreateColumnMajorLoad(&EltTy, &Matrix, Alignment, &Stride,
                                       /*isVolatile*/ false, ColsAsUInt64,
                                       RowsAsUInt64),
        ColsAsUInt64, RowsAsUInt64);

  return Vec;
}

// A helper function that stores column-major order Matrix into Dest as either
// column or row-major order depending on isColMajor value.
void storeFlatVectorMatrix(MatrixBuilder &MBuilder, Value &Matrix,
                           Value &Dest, Value &Rows, Value &Columns,
                           Value &Stride, bool IsColMajor,
                           const Align &Alignment) {
  uint64_t RowsAsUInt64 = cast<ConstantInt>(Rows).getZExtValue();
  uint64_t ColsAsUInt64 = cast<ConstantInt>(Columns).getZExtValue();
  if (IsColMajor)
    MBuilder.CreateColumnMajorStore(&Matrix, &Dest, Align(Alignment), &Stride,
                                    /*isVolatile*/ false, RowsAsUInt64,
                                    ColsAsUInt64);
  else
    MBuilder.CreateColumnMajorStore(
        MBuilder.CreateMatrixTranspose(&Matrix, RowsAsUInt64, ColsAsUInt64),
        &Dest, Align(Alignment), &Stride, /*isVolatile*/ false, ColsAsUInt64,
        RowsAsUInt64);
}

/// A helper function to retrieve the scalar type of a value pointer.
///
/// \param V a value pointer to a scalar type or a 2D array of scalar values
///
/// \returns the scalar type of a value pointer to 2D array or scalar pointed by
/// \p V.
Type *getMatrixElementType(const Value &V) {
  auto *ElementType = V.getType()->getPointerElementType();
  if (ElementType->isArrayTy()) {
    ElementType = ElementType->getArrayElementType();
    if (ElementType->isArrayTy())
      ElementType = ElementType->getArrayElementType();
  }
  assert(ElementType->isIntegerTy() || ElementType->isFloatingPointTy());
  return ElementType;
}

/// A helper function that Down-/uppercasts integer value to Int32
///
/// \param V a Value pointer to an integer type
/// \param downcast this value is set to true if the value \p V was downcast
///
/// \returns the downcast value
auto *prepBLASInt32(IRBuilder<> &IR, Value *V, bool &Downcast) {
  if (V->getType()->getIntegerBitWidth() > 32)
    Downcast |= true; // We're doing a potentially (but unlikely) illegal cast.
  if (V->getType()->getIntegerBitWidth() != 32)
    V = IR.CreateIntCast(V, IR.getInt32Ty(), false);
  return V;
}

/// A helper function that returns a constant scalar value of 1 if it was not
/// matched (alpha and beta implicitly equal to 1).
///
/// \param V a Value that points to a scalar or nullptr
/// \param opTy the Type of the scalar pointed by \p
///
/// \returns \p V if it not nullptr. Otherwise, returns a constant equal to 1.
Value *prepBLASScalar(IRBuilder<> &IR, Value *V, Type *OpTy, double Init = 1.) {
  Value *Scalar;
  if (OpTy->isFloatTy()) {
    float f = Init;
    Scalar = ConstantFP::get(OpTy, APFloat(f));
  } else if (OpTy->isDoubleTy()) {
    double d = Init;
    Scalar = ConstantFP::get(OpTy, APFloat(d));
  } else {
    llvm_unreachable("Scalar needs to be either FloatTy or DoubleTy.");
  }
  return V != nullptr ? V : Scalar;
}

/// A helper function that returns the base pointer to matrix \p M. If the
/// base pointer points to a vector of pointers it needs to be casted to a
/// flat-pointer before it is passed to cblas_X()
///
/// \param IR the IR builder handling the current function
/// transformation
///
/// \param M the matrix from which the base pointer will be returned
///
/// \returns the base pointer to matrix \p M.
auto *getFlatPointerToMatrix(IRBuilder<> &IR, const GEMMFaRer::Matrix &M) {
  // Flatten array
  auto *BasePtr = &M.getBaseAddressPointer();
  auto *DestTy = getMatrixElementType(*BasePtr)->getPointerTo();
  BasePtr = IR.CreateBitCast(BasePtr, DestTy);

  // If we have a pointer to a vector (e.g. [1024 x double]*) it is safe to
  // convert it to a pointer to the base type. We cast away explicit size
  // info but BLAS doesn't care.
  if (auto *MATy = dyn_cast<ArrayType>(getMatrixElementType(*BasePtr)))
    BasePtr = IR.CreatePointerCast(BasePtr,
                                   MATy->getArrayElementType()->getPointerTo());
  return BasePtr;
}

inline void insertNoInlineCall(Module &M, IRBuilder<> &IR,
                               ArrayRef<Type *> ArgTys, ArrayRef<Value *> Args,
                               StringRef FunctionName) {
  // Add a declaration for the function we're going to be replacing with.
  auto *FTy = FunctionType::get(IR.getVoidTy(), ArgTys, false);
  FunctionCallee F = M.getOrInsertFunction(FunctionName, FTy);

  // We can never inline this call.
  cast<Function>(F.getCallee())->addFnAttr(Attribute::NoInline);

  // Create the call to the function.
  CallInst *Call = IR.CreateCall(F, Args);
  Call->setIsNoInline();
}

void buildBLASGEMMCall(Module &Mod, IRBuilder<> &IR,
                       const GEMMFaRer::GEMM &Gemm) {

  const GEMMFaRer::Matrix &MA = Gemm.getMatrixA();
  const GEMMFaRer::Matrix &MB = Gemm.getMatrixB();
  const GEMMFaRer::Matrix &MC = Gemm.getMatrixC();

  // Matrix C's layout defines cblas_X() layout bacause it cannot be trasposed.
  ConstantInt *Layout = IR.getInt32(MC.getLayout());
  ConstantInt *TransA =
      IR.getInt32(MA.getLayout() == MC.getLayout() ? CBLAS_TRANSPOSE::NoTrans
                                                   : CBLAS_TRANSPOSE::Trans);
  ConstantInt *TransB =
      IR.getInt32(MB.getLayout() == MC.getLayout() ? CBLAS_TRANSPOSE::NoTrans
                                                   : CBLAS_TRANSPOSE::Trans);

  // BLAS interface only supports I32 so we will warn when we downcast.
  bool Downcast = false;

  // Make args for M, N, K.
  auto *M = prepBLASInt32(IR, &MA.getRows(), Downcast);
  auto *K = prepBLASInt32(IR, &MA.getColumns(), Downcast);
  auto *N = prepBLASInt32(IR, &MB.getColumns(), Downcast);

  // Args for memory pointers to A, B, C
  auto *A = getFlatPointerToMatrix(IR, MA);
  auto *B = getFlatPointerToMatrix(IR, MB);
  auto *C = getFlatPointerToMatrix(IR, MC);

  // Make args for LDA, LDB, LDC.
  auto *LDA = prepBLASInt32(IR, &MA.getLeadingDimensionSize(), Downcast);
  auto *LDB = prepBLASInt32(IR, &MB.getLeadingDimensionSize(), Downcast);
  auto *LDC = prepBLASInt32(IR, &MC.getLeadingDimensionSize(), Downcast);

  // C's pointed to type defines the operation type.
  auto *OpTy = getMatrixElementType(*C);

  // Make args for alpha/beta.
  Value *Alpha = prepBLASScalar(IR, Gemm.getAlpha(), OpTy);
  Value *Beta =
      prepBLASScalar(IR, Gemm.getBeta(), OpTy, Gemm.IsCReduced() ? 1.0 : 0.0);

  // Sanity type checking.
  assert(getMatrixElementType(*A) == OpTy && "A and C are typed differently.");
  assert(getMatrixElementType(*B) == OpTy && "B and C are typed differently.");
  assert(Alpha->getType() == OpTy && "Alpha and C are typed differently.");
  assert(Beta->getType() == OpTy && "Beta and C are typed differently.");

  // Send out downcast warning.
  if (Downcast)
    errs() << "A BLAS transform argument was larger than i32 and needed to be"
              " downcast.\nThis operation is potentially illegal.\n";

  // Prepare argument list
  auto *I32 = IR.getInt32Ty();
  auto *OpPtrTy = OpTy->getPointerTo();
  Type *ArgTys[] = {I32,     I32, I32,     I32, I32,  I32,     OpTy,
                    OpPtrTy, I32, OpPtrTy, I32, OpTy, OpPtrTy, I32};
  Value *Args[] = {Layout, TransA, TransB, M,   N,    K, Alpha,
                   A,      LDA,    B,      LDB, Beta, C, LDC};

  // Insert prepared call in the IR
  StringRef BlasFunctionName =
      OpTy == IR.getFloatTy() ? "cblas_sgemm" : "cblas_dgemm";
  insertNoInlineCall(Mod, IR, ArgTys, Args, BlasFunctionName);
}

// Adds a call to llvm.matrix.multiply to the IR
void buildMMIntrinsicCall(IRBuilder<> &IR, const GEMMFaRer::GEMM &Gemm) {

  const GEMMFaRer::Matrix &MA = Gemm.getMatrixA();
  const GEMMFaRer::Matrix &MB = Gemm.getMatrixB();
  const GEMMFaRer::Matrix &MC = Gemm.getMatrixC();

  // Get args for A, B, C.
  auto &A = MA.getBaseAddressPointer();
  auto &B = MB.getBaseAddressPointer();
  auto &C = MC.getBaseAddressPointer();

  // Matrix elements type checks
  auto *AElType = getMatrixElementType(A);
  auto *BElType = getMatrixElementType(B);
  auto *CElType = getMatrixElementType(C);
  assert(AElType == CElType && "A and C are typed differently.");
  assert(BElType == CElType && "B and C are typed differently.");

  // Get dimension sizes as i64 as expected by llvm.matrix.* intrinsics
  auto *M = IR.CreateSExt(&MA.getRows(), IR.getInt64Ty());
  auto *N = IR.CreateSExt(&MB.getColumns(), IR.getInt64Ty());
  auto *K = IR.CreateSExt(&MA.getColumns(), IR.getInt64Ty());

  assert(dyn_cast_or_null<ConstantInt>(M) && "M dimension is not a constant.");
  assert(dyn_cast_or_null<ConstantInt>(N) && "N dimension is not a constant.");
  assert(dyn_cast_or_null<ConstantInt>(K) && "K dimension is not a constant.");

  uint64_t MAsUInt64 = cast<ConstantInt>(M)->getZExtValue();
  uint64_t NAsUInt64 = cast<ConstantInt>(N)->getZExtValue();
  uint64_t KAsUInt64 = cast<ConstantInt>(K)->getZExtValue();

  // Get leading dimension sizes as i64 as expected by llvm.matrix.* intrinsics
  auto *LDA = IR.CreateSExt(&MA.getLeadingDimensionSize(), IR.getInt64Ty());
  auto *LDB = IR.CreateSExt(&MB.getLeadingDimensionSize(), IR.getInt64Ty());
  auto *LDC = IR.CreateSExt(&MC.getLeadingDimensionSize(), IR.getInt64Ty());

  // Helper matrix builder to create calls to llvm.matrix.* intrinsics
  MatrixBuilder MBuilder(IR);

  const unsigned BitsInAByte = 8;

  auto IsAColMajor = MA.getLayout() == GEMMFaRer::ColMajor;
  auto const &AAlign = Align(AElType->getPrimitiveSizeInBits() / BitsInAByte);
  auto IsBColMajor = MB.getLayout() == GEMMFaRer::ColMajor;
  auto const &BAlign = Align(BElType->getPrimitiveSizeInBits() / BitsInAByte);

  auto *PtrToMatrixElType = CElType->getPointerTo(/*AddressSpace*/ 0);
  auto &APtr = *IR.CreateBitCast(&A, PtrToMatrixElType);
  auto &BPtr = *IR.CreateBitCast(&B, PtrToMatrixElType);
  auto &CPtr = *IR.CreateBitCast(&C, PtrToMatrixElType);

  // Load A and B into flat vectors
  Value *AVec =
      loadMatrixToFlatVector(MBuilder, *AElType, APtr, *M, *K, *LDA, IsAColMajor, AAlign);
  Value *BVec =
      loadMatrixToFlatVector(MBuilder, *BElType, BPtr, *K, *N, *LDB, IsBColMajor, BAlign);

  // Call llvm.matrix.multiply.*
  Value *CVec = MBuilder.CreateMatrixMultiply(AVec, BVec, MAsUInt64, KAsUInt64,
                                              NAsUInt64);

  auto *Alpha = Gemm.getAlpha();
  auto *Beta = Gemm.getBeta();

  Value *NewC = CVec;
  if (Alpha != nullptr)
    NewC = MBuilder.CreateScalarMultiply(Alpha, CVec);

  auto IsCColMajor = MC.getLayout() == GEMMFaRer::ColMajor;
  auto const &CAlign = Align(BElType->getPrimitiveSizeInBits() / BitsInAByte);

  if (Gemm.IsCReduced()) {
    if (Beta == nullptr)
      Beta = ConstantFP::get(CElType, APFloat(1.));
    NewC = MBuilder.CreateAdd(
        NewC, MBuilder.CreateScalarMultiply(
                  Beta, loadMatrixToFlatVector(MBuilder, *CElType, CPtr, *M, *N,
                                               *LDC, IsCColMajor, CAlign)));
  }

  // Store result into C
  storeFlatVectorMatrix(MBuilder, *NewC, CPtr, *M, *N, *LDC, IsCColMajor,
                        CAlign);
}

} // End anonymous namespace.

namespace GEMMFaRer {

// Replaces the corresponding basic blocks of MatMul IR with a call to
// llvm.matrix.multiply.
bool runImpl(Function &F, GEMMMatcher::Result &GMPR) {
  // Have we changed in this function?
  bool Changed = false;

  // Iterate through all GEMMs found.
  for (auto &GEMM : *GMPR) {
    if (!GEMMDataAnalysisPass::run(GEMM)) {
      LLVM_DEBUG(dbgs() << "GEMM not rewritable at line "
                        << GEMM.getAssociatedLoop().getStartLoc().getLine()
                        << '\n');
      continue;
    }
    LLVM_DEBUG(dbgs() << "GEMM rewritable at line "
                      << GEMM.getAssociatedLoop().getStartLoc().getLine()
                      << '\n');
    // Get the loop associated with this GEMM.
    Loop &L = GEMM.getAssociatedLoop();

    // We can't transform two dimensional pointers.
    auto *ATy = GEMM.getMatrixA().getBaseAddressPointer().getType();
    auto *BTy = GEMM.getMatrixB().getBaseAddressPointer().getType();
    auto *CTy = GEMM.getMatrixC().getBaseAddressPointer().getType();
    if (ATy->isPointerTy() && ATy->getPointerElementType()->isPointerTy()) {
      LLVM_DEBUG(dbgs() << "Matrix A was two dimensional pointer.\n");
      return false;
    }
    if (BTy->isPointerTy() && BTy->getPointerElementType()->isPointerTy()) {
      LLVM_DEBUG(dbgs() << "Matrix B was two dimensional pointer.\n");
      return false;
    }
    if (CTy->isPointerTy() && CTy->getPointerElementType()->isPointerTy()) {
      LLVM_DEBUG(dbgs() << "Matrix C was two dimensional pointer.\n");
      return false;
    }

    // Collect the only possible block exited *from*. This was verified already
    // in the GEMMMatcherPass but we're asserting here as a sanity check.
    BasicBlock *ExitingBlock = L.getExitingBlock();
    assert(ExitingBlock != nullptr && "Loop had multiple exiting blocks");

    // Collect the only one possible block exited *to*. This was verified
    // already in the GEMMMatcherPass but we're asserting here as a sanity
    // check.
    BasicBlock *ExitBlock = L.getExitBlock();
    assert(ExitBlock != nullptr && "Loop had multiple exit blocks.");

    // Make the call to llvm.matrix.multiply
    IRBuilder<> IR(ExitBlock, ExitBlock->getFirstInsertionPt());

    if (ReplaceMode == GEMMFaRer::MatrixIntrinsics)
      // Make the call to llvm.matrix.multiply
      buildMMIntrinsicCall(IR, GEMM);
    else if (ReplaceMode == GEMMFaRer::CBLAS)
      // Make the call using the CBLAS interface
      buildBLASGEMMCall(*F.getParent(), IR, GEMM);
    else
      assert(0 && "Unknown GeMM replacement mode!");

    // Delete reduction store to Matrix C, thus making the reduction dead-code
    GEMM.getReductionStore().eraseFromParent();

    // Mark that we changed here.
    Changed = true;
  }

  return Changed;
}

} // end of namespace GEMMFaRer

/// Replaces Matrix Multiply occurencies with calls to llvm.matrix.multiply
// Runs on each function, makes a list of candidates and updates their IR when
// the change is possible
PreservedAnalyses GEMMReplacerPass::run(Function &F,
                                        FunctionAnalysisManager &FAM) {
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  GEMMFaRer::GEMMMatcher::Result GMPR = GEMMFaRer::GEMMMatcher::run(F, LI, DT);
  bool Changed = runImpl(F, GMPR);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  // TODO: add here what we *do* preserve.
  return PA;
}

#undef DEBUG_TYPE
