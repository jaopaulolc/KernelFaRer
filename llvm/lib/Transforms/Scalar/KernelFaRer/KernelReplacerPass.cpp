//===- KernelReplacer.cpp - Kernel Replacer Pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass replaces GEMM or SYR2K loops previously recognized by the matcher
// into CBLAS or EIGEN calls. In cases that this kicks in, it can be a
// significant performance win.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/KernelFaRer.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "kernel-replacer-pass"

using namespace llvm;
using namespace KernelFaRer;

// Anonymous namespace containing rewriter functions.
namespace {

// Alpha and Beta can be passed through CLI for the cases when the matcher does
// not yet match them
cl::opt<double> AlphaInit("alpha", cl::desc("alpha"), cl::ValueRequired,
                          cl::init(1.0));
cl::opt<double> BetaInit("beta", cl::desc("beta"), cl::ValueRequired,
                         cl::init(1.0));

// Command line argument that determines which mode we are replacing in.
cl::opt<KernelFaRer::ReplacementMode> ReplaceMode(
    "kernelfarer-replacement-mode",
    cl::desc("Available kernel replacement methods."),
    cl::values(clEnumValN(KernelFaRer::CBLAS, "cblas-interface",
                          "Replace using the CBLAS interface."),
               clEnumValN(KernelFaRer::EIGEN, "eigen-runtime",
                          "Replace using the eigen runtime interface.")),
    cl::ValueRequired, cl::init(KernelFaRer::UNKNOWN));

// Constants are from Eigen enum values.
// https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/util/Constants.h#L316
constexpr int32_t EigenColMaj = 0;
constexpr int32_t EigenRowMaj = 1;

constexpr unsigned int EigenMaxArgs = 14;
constexpr unsigned int EigenMaxFNameLen = 13;
constexpr size_t EigenSizeWidth = 64;


/// A helper function to retrieve the scalar type of a value pointer.
///
/// \param M a value pointer to a scalar type or a 2D array of scalar values
///
/// \returns the scalar type of a value pointer to 2D array or scalar pointed by
/// \p M.
Type *getMatrixElementType(const Value &M) {
  auto *ElementType = M.getType()->getPointerElementType();
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
Value *prepScalar(IRBuilder<> &IR, Value *V, Type *OpTy, double Init = 1.) {
  Value *Scalar;
  if (OpTy->isFloatTy())
    Scalar = ConstantFP::get(OpTy, APFloat((float)Init));
  else if (OpTy->isDoubleTy())
    Scalar = ConstantFP::get(OpTy, APFloat((double)Init));
  else
    llvm_unreachable("Scalar needs to be either FloatTy or DoubleTy.");
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
auto *getFlatPointerToMatrix(IRBuilder<> &IR, const KernelFaRer::Matrix &M) {
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
                       const KernelFaRer::GEMM &Gemm) {

  const KernelFaRer::Matrix &MatA = Gemm.getMatrixA();
  const KernelFaRer::Matrix &MatB = Gemm.getMatrixB();
  const KernelFaRer::Matrix &MatC = Gemm.getMatrixC();

  // Matrix C's layout defines cblas_X() layout bacause it cannot be trasposed.
  ConstantInt *Layout = IR.getInt32(MatC.getLayout());
  ConstantInt *TransA =
      IR.getInt32(MatA.getLayout() == MatC.getLayout() ? CBLAS_TRANSPOSE::NoTrans
                                                   : CBLAS_TRANSPOSE::Trans);
  ConstantInt *TransB =
      IR.getInt32(MatB.getLayout() == MatC.getLayout() ? CBLAS_TRANSPOSE::NoTrans
                                                   : CBLAS_TRANSPOSE::Trans);

  // BLAS interface only supports I32 so we will warn when we downcast.
  bool Downcast = false;

  // Make args for M, N, K.
  auto *M = prepBLASInt32(IR, &MatA.getRows(), Downcast);
  auto *K = prepBLASInt32(IR, &MatA.getColumns(), Downcast);
  auto *N = prepBLASInt32(IR, &MatB.getColumns(), Downcast);

  // Args for memory pointers to A, B, C
  auto *A = getFlatPointerToMatrix(IR, MatA);
  auto *B = getFlatPointerToMatrix(IR, MatB);
  auto *C = getFlatPointerToMatrix(IR, MatC);

  // Make args for LDA, LDB, LDC.
  auto *LDA = prepBLASInt32(IR, &MatA.getLeadingDimensionSize(), Downcast);
  auto *LDB = prepBLASInt32(IR, &MatB.getLeadingDimensionSize(), Downcast);
  auto *LDC = prepBLASInt32(IR, &MatC.getLeadingDimensionSize(), Downcast);

  // C's pointed to type defines the operation type.
  auto *OpTy = getMatrixElementType(*C);

  // Make args for alpha/beta.
  Value *Alpha = prepScalar(IR, Gemm.getAlpha(), OpTy);
  Value *Beta =
      prepScalar(IR, Gemm.getBeta(), OpTy, Gemm.isCReduced() ? 1.0 : 0.0);

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
  auto *A = getFlatPointerToMatrix(IR, MA);
  auto *B = getFlatPointerToMatrix(IR, MB);
  auto *C = getFlatPointerToMatrix(IR, MC);

  // C's pointed to type defines the operation type.
  Type *opTy = getMatrixElementType(*C);

  // Make args for LDA, LDB, LDC
  auto *LDA = prepBLASInt32(IR, &MA.getLeadingDimensionSize(), Downcast);
  auto *LDB = prepBLASInt32(IR, &MB.getLeadingDimensionSize(), Downcast);
  auto *LDC = prepBLASInt32(IR, &MC.getLeadingDimensionSize(), Downcast);

  // Send out downcast warning.
  if (Downcast)
    errs() << "A BLAS transform argument was larger than i32 and needed to be"
              " downcast.\nThis operation is potentially illegal.\n";

  // Make args for alpha/beta.
  auto *Alpha = prepScalar(IR, Syr2k.getAlpha(), opTy, AlphaInit);
  auto *Beta = prepScalar(IR, Syr2k.getBeta(), opTy, BetaInit);

  // Sanity type checking.
  assert(getMatrixElementType(*A) == opTy && "A and C are typed differently.");
  assert(getMatrixElementType(*B) == opTy && "B and C are typed differently.");
  assert(Alpha->getType() == opTy && "Alpha and C are typed differently.");
  assert(Beta->getType() == opTy && "Beta and C are typed differently.");

  // Prepare argument list
  IntegerType *I32 = IR.getInt32Ty();
  Type *opPtrTy = opTy->getPointerTo();
  Type *ArgTys[] = {I32, I32,     I32, I32,  I32,     opTy, opPtrTy,
                    I32, opPtrTy, I32, opTy, opPtrTy, I32};
  Value *Args[] = {Layout, Uplo, Trans, N,    K, Alpha, A,
                   LDA,    B,    LDB,   Beta, C, LDC};

  // Insert prepared call in the IR
  StringRef FunctionName =
      (opTy == IR.getFloatTy()) ? "cblas_ssyr2k" : "cblas_dsyr2k";
  insertNoInlineCall(Mod, IR, ArgTys, Args, FunctionName);
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
  auto *A = getFlatPointerToMatrix(IR, MA);
  auto *B = getFlatPointerToMatrix(IR, MB);
  auto *C = getFlatPointerToMatrix(IR, MC);

  // C's pointed to type defines the operation type.
  Type *OpTy = getMatrixElementType(*C);

  // Make args for alpha/beta.
  Value *Alpha = Gemm.getAlpha();
  Value *Beta = Gemm.getBeta();

  // Sanity type checking.
  assert(getMatrixElementType(*A) == OpTy && "A and C are typed differently.");
  assert(getMatrixElementType(*B) == OpTy && "B and C are typed differently.");

  // Add args and types for A, B, C.
  Args.emplace_back(A);
  Args.emplace_back(B);
  Args.emplace_back(C);
  for (size_t I = 0; I < 3; ++I) {
    ArgTys.emplace_back(OpTy->getPointerTo());
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
bool runImpl(Function &F, LoopInfo &LI, DominatorTree &DT, KernelMatcher::Result &KMPR) {
  // Have we changed in this function?
  bool Changed = false;

  // Iterate through all GEMMs found.
  for (std::unique_ptr<Kernel> &Ker : *KMPR) {
    if (!KernelDataAnalysisPass::run(*Ker)) {
      LLVM_DEBUG(dbgs() << "Kernel not rewritable at line "
                        << Ker->getAssociatedLoop().getStartLoc().getLine()
                        << '\n');
      continue;
    }
    LLVM_DEBUG(dbgs() << "Kernel rewritable at line "
                      << Ker->getAssociatedLoop().getStartLoc().getLine()
                      << '\n');
    // Get the loop associated with this GEMM.
    Loop &L = Ker->getAssociatedLoop();

    // We can't transform two dimensional pointers.
    auto *ATy = Ker->getMatrixA().getBaseAddressPointer().getType();
    auto *BTy = Ker->getMatrixB().getBaseAddressPointer().getType();
    auto *CTy = Ker->getMatrixC().getBaseAddressPointer().getType();
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
    // in the KernelMatcherPass but we're asserting here as a sanity check.
    BasicBlock *ExitingBlock = L.getExitingBlock();
    assert(ExitingBlock != nullptr && "Loop had multiple exiting blocks");

    // Collect the only one possible block exited *to*. This was verified
    // already in the KernelMatcherPass but we're asserting here as a sanity
    // check.
    BasicBlock *ExitBlock = L.getExitBlock();
    assert(ExitBlock != nullptr && "Loop had multiple exit blocks.");

    IRBuilder<> IR(ExitBlock, ExitBlock->getFirstInsertionPt());

    if (ReplaceMode == KernelFaRer::CBLAS) {
      // Make the call using the CBLAS interface
      if (const auto *GEMM = dyn_cast_or_null<KernelFaRer::GEMM>(Ker.get()))
        buildBLASGEMMCall(*F.getParent(), IR, *GEMM);
      if (const auto *SYR2K = dyn_cast_or_null<KernelFaRer::SYR2K>(Ker.get()))
        buildBLASSYR2KCall(*F.getParent(), IR, *SYR2K);
    } else if (ReplaceMode == KernelFaRer::EIGEN) {
      // Make the call using Eigen runtime interface
      if (const auto *GEMM = dyn_cast_or_null<KernelFaRer::GEMM>(Ker.get()))
        buildEigenCall(*F.getParent(), IR, *GEMM);
      else
        assert(0 && "Eigen runtime only supports GEMM");
    } else
      assert(0 && "Unknown GeMM replacement mode!");

    // Delete reduction store to Matrix C, thus making the reduction dead-code
    Ker->getReductionStore().eraseFromParent();

    // Mark that we changed here.
    Changed = true;
  }

  return Changed;
}

} // end of namespace KernelFaRer

/// Replaces Kernel occurencies with calls to CBLAS or EIGEN
// Runs on each function, makes a list of candidates and updates their IR when
// the change is possible
PreservedAnalyses KernelReplacerPass::run(Function &F,
                                        FunctionAnalysisManager &FAM) {
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  KernelFaRer::KernelMatcher::Result KMPR = KernelFaRer::KernelMatcher::run(F, LI, DT);
  bool Changed = runImpl(F, LI, DT, KMPR);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  // TODO: add here what we *do* preserve.
  return PA;
}

#undef DEBUG_TYPE
