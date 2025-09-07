//===- KernelReplacer.cpp - Kernel Replacer Pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass replaces matrix-multiply or syr2k loops previously recognized by
// the matcher into either llvm.matrix.multiply.* or high-performance library
// intrinsic calls.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "KernelFaRer.h"

#define PLUGIN_NAME "kernelfarer"
#define PASS_NAME "kernel-replacer-pass"

using namespace llvm;
using namespace KernelFaRer;

// Anonymous namespace containing rewriter functions.
namespace {

// Alpha and Beta can be passed through CLI for the cases when the matcher does
// not yet match them
static cl::opt<double> AlphaInit("alpha", cl::desc("alpha"), cl::ValueRequired,
                          cl::init(1.0));
static cl::opt<double> BetaInit("beta", cl::desc("beta"), cl::ValueRequired,
                         cl::init(1.0));

// Command line argument that determines which mode we are replacing in.
static cl::opt<KernelFaRer::ReplacementMode> ReplaceMode(
    "kernelfarer-replacement-mode",
    cl::desc("Available kernel replacement methods."),
    cl::values(clEnumValN(KernelFaRer::MatrixIntrinsics, "matrix-intrinsics",
                          "Replace using llvm.matrix.* intrinsics."),
               clEnumValN(KernelFaRer::CBLAS, "cblas-interface",
                          "Replace using the CBLAS interface."),
               clEnumValN(KernelFaRer::EIGEN, "eigen-runtime",
                          "Replace using the eigen runtime interface.")),
    cl::ValueRequired, cl::init(KernelFaRer::CBLAS));

// Constants are from Eigen enum values.
// https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/util/Constants.h#L316
constexpr int32_t EigenColMaj = 0;
constexpr int32_t EigenRowMaj = 1;

constexpr unsigned int EigenMaxArgs = 14;
constexpr unsigned int EigenMaxFNameLen = 13;
constexpr size_t EigenSizeWidth = 64;

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
void storeFlatVectorMatrix(MatrixBuilder &MBuilder, Value &Matrix, Value &Dest,
                           Value &Rows, Value &Columns, Value &Stride,
                           bool IsColMajor, const Align &Alignment) {
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

/// A helper function that uppercasts integer value to Int64 if needed
///
/// \param V a Value pointer to an integer value
///
/// \returns \p V or the uppercasted \p V
auto *prepEigenInt64(IRBuilder<> &IR, Value *const &V) {
  if (!V->getType()->isIntegerTy(EigenSizeWidth)) {
    return IR.CreateIntCast(V, IR.getIntNTy(EigenSizeWidth), false);
  }
  return V;
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
                       const KernelFaRer::GEMM &Gemm) {

  const KernelFaRer::Matrix &MA = Gemm.getMatrixA();
  const KernelFaRer::Matrix &MB = Gemm.getMatrixB();
  const KernelFaRer::Matrix &MC = Gemm.getMatrixC();

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
  auto *A = &MA.getBaseAddressPointer();
  auto *B = &MB.getBaseAddressPointer();
  auto *C = &MC.getBaseAddressPointer();

  // Make args for LDA, LDB, LDC.
  auto *LDA = prepBLASInt32(IR, &MA.getLeadingDimensionSize(), Downcast);
  auto *LDB = prepBLASInt32(IR, &MB.getLeadingDimensionSize(), Downcast);
  auto *LDC = prepBLASInt32(IR, &MC.getLeadingDimensionSize(), Downcast);

  // C's pointed to type defines the operation type.
  auto *OpTy = MC.getScalarElementType();

  // Make args for alpha/beta.
  Value *Alpha = prepBLASScalar(IR, Gemm.getAlpha(), OpTy);
  Value *Beta =
      prepBLASScalar(IR, Gemm.getBeta(), OpTy, Gemm.IsCReduced() ? 1.0 : 0.0);

  // Sanity type checking.
  assert(MA.getScalarElementType() == OpTy && "A and C are typed differently.");
  assert(MB.getScalarElementType() == OpTy && "B and C are typed differently.");
  assert(Alpha->getType() == OpTy && "Alpha and C are typed differently.");
  assert(Beta->getType() == OpTy && "Beta and C are typed differently.");

  // Send out downcast warning.
  if (Downcast)
    errs() << "A BLAS transform argument was larger than i32 and needed to be"
              " downcast.\nThis operation is potentially illegal.\n";

  // Prepare argument list
  auto *I32 = IR.getInt32Ty();
  auto *OpPtrTy = PointerType::get(OpTy, /*AddressSpace*/ 0);
  Type *ArgTys[] = {I32,     I32, I32,     I32, I32,  I32,     OpTy,
                    OpPtrTy, I32, OpPtrTy, I32, OpTy, OpPtrTy, I32};
  Value *Args[] = {Layout, TransA, TransB, M,   N,    K, Alpha,
                   A,      LDA,    B,      LDB, Beta, C, LDC};

  // Insert prepared call in the IR
  StringRef BlasFunctionName =
      OpTy == IR.getFloatTy() ? "cblas_sgemm" : "cblas_dgemm";
  insertNoInlineCall(Mod, IR, ArgTys, Args, BlasFunctionName);
}

void buildBLASSYR2KCall(Module &Mod, IRBuilder<> &IR, const Kernel &Syr2k) {
  // Saves function calls and characters
  const Matrix &MA = Syr2k.getMatrixA();
  const Matrix &MB = Syr2k.getMatrixB();
  const Matrix &MC = Syr2k.getMatrixC();

  // Options
  ConstantInt *Layout = IR.getInt32(RowMajor);
  ConstantInt *Uplo = IR.getInt32(Lower);
  ConstantInt *Trans = IR.getInt32(NoTrans);

  // BLAS interface only supports I32 so we will warn when we downcast.
  bool Downcast = false;

  // Dimension
  auto *N = prepBLASInt32(IR, &MA.getRows(), Downcast);
  auto *K = prepBLASInt32(IR, &MA.getColumns(), Downcast);

  // Args for memory pointers to A, B, C
  auto *A = &MA.getBaseAddressPointer();
  auto *B = &MB.getBaseAddressPointer();
  auto *C = &MC.getBaseAddressPointer();

  // C's pointed to type defines the operation type.
  Type *OpTy = MC.getScalarElementType();

  // Make args for LDA, LDB, LDC
  auto *LDA = prepBLASInt32(IR, &MA.getLeadingDimensionSize(), Downcast);
  auto *LDB = prepBLASInt32(IR, &MB.getLeadingDimensionSize(), Downcast);
  auto *LDC = prepBLASInt32(IR, &MC.getLeadingDimensionSize(), Downcast);

  // Send out downcast warning.
  if (Downcast)
    errs() << "A BLAS transform argument was larger than i32 and needed to be"
              " downcast.\nThis operation is potentially illegal.\n";

  // Make args for alpha/beta.
  auto *Alpha = prepBLASScalar(IR, Syr2k.getAlpha(), OpTy, AlphaInit);
  auto *Beta = prepBLASScalar(IR, Syr2k.getBeta(), OpTy, BetaInit);

  // Sanity type checking.
  assert(MA.getScalarElementType() == OpTy && "A and C are typed differently.");
  assert(MB.getScalarElementType() == OpTy && "B and C are typed differently.");
  assert(Alpha->getType() == OpTy && "Alpha and C are typed differently.");
  assert(Beta->getType() == OpTy && "Beta and C are typed differently.");

  // Prepare argument list
  IntegerType *I32 = IR.getInt32Ty();
  auto *OpPtrTy = PointerType::get(OpTy, /*AddressSpace*/ 0);
  Type *ArgTys[] = {I32, I32,     I32, I32,  I32,     OpTy, OpPtrTy,
                    I32, OpPtrTy, I32, OpTy, OpPtrTy, I32};
  Value *Args[] = {Layout, Uplo, Trans, N,    K, Alpha, A,
                   LDA,    B,    LDB,   Beta, C, LDC};

  // Insert prepared call in the IR
  StringRef FunctionName =
      (OpTy == IR.getFloatTy()) ? "cblas_ssyr2k" : "cblas_dsyr2k";
  insertNoInlineCall(Mod, IR, ArgTys, Args, FunctionName);
}

// Adds a call to llvm.matrix.multiply to the IR
void buildMMIntrinsicCall(IRBuilder<> &IR, const KernelFaRer::GEMM &Gemm) {

  const KernelFaRer::Matrix &MA = Gemm.getMatrixA();
  const KernelFaRer::Matrix &MB = Gemm.getMatrixB();
  const KernelFaRer::Matrix &MC = Gemm.getMatrixC();

  // Get args for A, B, C.
  auto &A = MA.getBaseAddressPointer();
  auto &B = MB.getBaseAddressPointer();
  auto &C = MC.getBaseAddressPointer();

  // Matrix elements type checks
  auto *AElType = MA.getScalarElementType();
  auto *BElType = MB.getScalarElementType();
  auto *CElType = MC.getScalarElementType();
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

  auto IsAColMajor = MA.getLayout() == KernelFaRer::ColMajor;
  auto const &AAlign = Align(AElType->getPrimitiveSizeInBits() / BitsInAByte);
  auto IsBColMajor = MB.getLayout() == KernelFaRer::ColMajor;
  auto const &BAlign = Align(BElType->getPrimitiveSizeInBits() / BitsInAByte);

  auto *PtrToMatrixElType = PointerType::get(CElType, /*AddressSpace*/0);
  auto &APtr = *IR.CreateBitCast(&A, PtrToMatrixElType);
  auto &BPtr = *IR.CreateBitCast(&B, PtrToMatrixElType);
  auto &CPtr = *IR.CreateBitCast(&C, PtrToMatrixElType);

  // Load A and B into flat vectors
  Value *AVec = loadMatrixToFlatVector(MBuilder, *AElType, APtr, *M, *K, *LDA,
                                       IsAColMajor, AAlign);
  Value *BVec = loadMatrixToFlatVector(MBuilder, *BElType, BPtr, *K, *N, *LDB,
                                       IsBColMajor, BAlign);

  // Call llvm.matrix.multiply.*
  Value *CVec = MBuilder.CreateMatrixMultiply(AVec, BVec, MAsUInt64, KAsUInt64,
                                              NAsUInt64);

  auto *Alpha = Gemm.getAlpha();
  auto *Beta = Gemm.getBeta();

  Value *NewC = CVec;
  if (Alpha != nullptr)
    NewC = MBuilder.CreateScalarMultiply(Alpha, CVec);

  auto IsCColMajor = MC.getLayout() == KernelFaRer::ColMajor;
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

// Adds the call to Eigen runtime
void buildEigenCall(Module &Mod, IRBuilder<> &IR, const KernelFaRer::GEMM &Gemm) {

  const KernelFaRer::Matrix &MA = Gemm.getMatrixA();
  const KernelFaRer::Matrix &MB = Gemm.getMatrixB();
  const KernelFaRer::Matrix &MC = Gemm.getMatrixC();

  // The vector of arguments.
  SmallVector<Value *, EigenMaxArgs> Args;
  SmallVector<Type *, EigenMaxArgs> ArgTys;

  // Get args for A, B, C.
  auto *A = &MA.getBaseAddressPointer();
  auto *B = &MB.getBaseAddressPointer();
  auto *C = &MC.getBaseAddressPointer();

  // C's pointed to type defines the operation type.
  Type *OpTy = MC.getScalarElementType();

  // Make args for alpha/beta.
  Value *Alpha = Gemm.getAlpha();
  Value *Beta = Gemm.getBeta();

  // Sanity type checking.
  assert(MA.getScalarElementType() == OpTy && "A and C are typed differently.");
  assert(MB.getScalarElementType() == OpTy && "B and C are typed differently.");

  // Add args and types for A, B, C.
  Args.emplace_back(A);
  Args.emplace_back(B);
  Args.emplace_back(C);
  for (size_t I = 0; I < 3; ++I) {
    ArgTys.emplace_back(PointerType::get(OpTy, /*AddressSpace*/0));
  }

  // Make the function name
  SmallString<EigenMaxFNameLen> FunctionName("_gemm");

  // Check for and add alpha/beta and types.
  if (Alpha != nullptr) {
    FunctionName += "A";
    Args.push_back(Alpha);
    ArgTys.push_back(OpTy);
    assert(Alpha->getType() == OpTy && "Alpha and C are typed differently.");
  }
  if (Beta != nullptr) {
    FunctionName += "B";
    Args.push_back(Beta);
    ArgTys.push_back(OpTy);
    assert(Beta->getType() == OpTy && "Beta and C are typed differently.");
  }

  // Finish the name.
  if (OpTy->isIntegerTy()) {
    FunctionName += "Uint";
    FunctionName += std::to_string(OpTy->getIntegerBitWidth());
  } else if (OpTy->isFloatTy()) {
    FunctionName += "Float";
  } else if (OpTy->isDoubleTy()) {
    FunctionName += "Double";
  } else {
    llvm_unreachable("Unknown opTy for Eigen.");
  }

  // Make and add args and types for layout*.
  bool ARowMajor = MA.getLayout() == KernelFaRer::RowMajor;
  bool BRowMajor = MB.getLayout() == KernelFaRer::RowMajor;
  bool CRowMajor = MC.getLayout() == KernelFaRer::RowMajor;
  Args.emplace_back(IR.getInt32(ARowMajor ? EigenRowMaj : EigenColMaj));
  Args.emplace_back(IR.getInt32(BRowMajor ? EigenRowMaj : EigenColMaj));
  Args.emplace_back(IR.getInt32(CRowMajor ? EigenRowMaj : EigenColMaj));
  for (size_t I = 0; I < 3; ++I) {
    ArgTys.emplace_back(IR.getInt32Ty());
  }

  // Make args for M, N, K.
  auto *M = prepEigenInt64(IR, &MA.getRows());
  auto *N = prepEigenInt64(IR, &MB.getColumns());
  auto *K = prepEigenInt64(IR, &MA.getColumns());

  // Add args and types for M, N, K.
  Args.emplace_back(M);
  Args.emplace_back(N);
  Args.emplace_back(K);
  for (size_t I = 0; I < 3; ++I) {
    ArgTys.emplace_back(IR.getIntNTy(EigenSizeWidth));
  }

  // Make args for LDA, LDB, LDC.
  auto *LDA = prepEigenInt64(IR, &MA.getLeadingDimensionSize());
  auto *LDB = prepEigenInt64(IR, &MB.getLeadingDimensionSize());
  auto *LDC = prepEigenInt64(IR, &MC.getLeadingDimensionSize());

  // Add args and types for LDA, LDB, LDC.
  Args.emplace_back(LDA);
  Args.emplace_back(LDB);
  Args.emplace_back(LDC);
  for (size_t I = 0; I < 3; ++I) {
    ArgTys.emplace_back(IR.getIntNTy(EigenSizeWidth));
  }

  insertNoInlineCall(Mod, IR, ArgTys, Args, FunctionName.str());
}

} // End anonymous namespace.

namespace KernelFaRer {

// Replaces the corresponding basic blocks of MatMul IR with a call to
// llvm.matrix.multiply.
bool runImpl(Function &F, KernelMatcher::Result &GMPR, OptimizationRemarkEmitter &ORE) {
  // Have we changed in this function?
  bool Changed = false;

  // Iterate through all GEMMs found.
  for (std::unique_ptr<Kernel> &Ker : *GMPR) {
    // Get the loop associated with this Kernel.
    Loop &L = Ker->getAssociatedLoop();
    auto DL = L.getStartLoc();
    auto ORM =
        OptimizationRemarkMissed(PASS_NAME, "MatcherRemark", DL, L.getHeader());
    auto ORA = OptimizationRemarkAnalysis(PASS_NAME, "MatcherRemark", DL,
                                          L.getHeader());
    if (!DataAnalysisPass::run(*Ker)) {
      ORE.emit(ORM << "Found Kernel is not rewritable!");
      continue;
    }
    ORE.emit(ORA << "Found Kernel is rewritable!");

    // We can't transform two dimensional pointers.
    if (Ker->getMatrixA().isDoublePtr()) {
      ORE.emit(ORM << "Matrix A is two dimensional pointer. Cannot replace kernel code.\n");
      return false;
    }
    if (Ker->getMatrixB().isDoublePtr()) {
      ORE.emit(ORM << "Matrix B was two dimensional pointer. Cannot replace kernel code.\n");
      return false;
    }
    if (Ker->getMatrixC().isDoublePtr()) {
      ORE.emit(ORM << "Matrix C was two dimensional pointer. Cannot replace kernel code.\n");
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

    IRBuilder<> IR(ExitBlock, ExitBlock->getFirstInsertionPt());

    if (ReplaceMode == KernelFaRer::MatrixIntrinsics)
      // Make the call to llvm.matrix.multiply
      if (const auto *GEMM = dyn_cast_or_null<KernelFaRer::GEMM>(Ker.get()))
        buildMMIntrinsicCall(IR, *GEMM);
      else
        assert(0 && "Intrinsic only handles GEMM!");
    else if (ReplaceMode == KernelFaRer::CBLAS) {
      // Make the call using the CBLAS interface
      if (const auto *GEMM = dyn_cast_or_null<KernelFaRer::GEMM>(Ker.get()))
        buildBLASGEMMCall(*F.getParent(), IR, *GEMM);
      if (const auto *SYR2K = dyn_cast_or_null<KernelFaRer::SYR2K>(Ker.get()))
        buildBLASSYR2KCall(*F.getParent(), IR, *SYR2K);
    } else if (ReplaceMode == KernelFaRer::EIGEN)
      // Make the call using Eigen runtime interface
      if (const auto *GEMM = dyn_cast_or_null<KernelFaRer::GEMM>(Ker.get()))
        buildEigenCall(*F.getParent(), IR, *GEMM);
      else
        assert(0 && "Intrinsic only handles GEMM!");
    else
      assert(0 && "Unknown GeMM replacement mode!");

    // Delete reduction store to Matrix C, thus making the reduction dead-code
    Ker->getReductionStore().eraseFromParent();

    // Mark that we changed here.
    Changed = true;
  }

  return Changed;
}

} // end of namespace KernelFaRer

/// Replaces Matrix Multiply or SYR2K occurencies with calls to
/// llvm.matrix.multiply intrinsic or high-performance libraries.
/// Runs on each function, makes a list of candidates and updates their IR when
/// the change is legal
PreservedAnalyses KernelReplacerPass::run(Function &F,
                                        FunctionAnalysisManager &FAM) {
  OptimizationRemarkEmitter &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  KernelFaRer::KernelMatcher::Result GMPR = KernelFaRer::KernelMatcher::run(F, LI, DT, ORE);
  bool Changed = runImpl(F, GMPR, ORE);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  // TODO: add here what we *do* preserve.
  return PA;
}

static void registerPasses(PassBuilder &PB) {
  PB.registerPipelineParsingCallback(
      [](StringRef Name, FunctionPassManager &FPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "kernel-replacer-pass") {
          FPM.addPass(KernelReplacerPass());
          return true;
        }
        return false;
      });
  PB.registerVectorizerStartEPCallback(
      [](FunctionPassManager &FPM, OptimizationLevel Level) {
        FPM.addPass(KernelReplacerPass());
      });
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PLUGIN_NAME, "0.1", registerPasses};
}

#undef DEBUG_TYPE
