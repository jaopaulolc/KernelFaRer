//===- Matcher.cpp - Kernel Recognition Pass ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements an idiom recognizer that identifies matrix-multiply and
// syr2kin normalized loops. This pass is used by the KernelReplacerPass to
// identify candidates that can be replaced by calls to high-performance
// libraries or llvm.matrix.multiply.* intrinsics.
//
//===----------------------------------------------------------------------===//
//
// TODO List:
//
// Detect matrices with unknown sizes.
// Support other data-types besides single and double precision floating-point.
// Properly support fmuladd intrinsic recognition.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"

#include "KernelFaRer.h"

using namespace llvm;
using namespace llvm::PatternMatch;
using namespace KernelFaRer;

#define PASS_NAME "kernel-matcher"

namespace llvm {
// Below novel and extended versions of matchers from PatternMatch are provided.
namespace PatternMatch {

// Provide a nonconst and const version as in PatternMatch.h.
/// Match a phi, capturing it if we match.
inline PatternMatch::bind_ty<PHINode> m_PHI(PHINode *&PHI) { return PHI; }
inline PatternMatch::bind_ty<const PHINode> m_PHI(const PHINode *&PHI) {
  return PHI;
}

/// Class that matches a phi's 1st and 2nd incoming values.
template <typename InVal1Ty, typename InVal2Ty> struct PHI_match {
  InVal1Ty InVal1;
  InVal2Ty InVal2;

  PHI_match(const InVal1Ty &IV1, const InVal2Ty &IV2)
      : InVal1(IV1), InVal2(IV2) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *PHI = dyn_cast<PHINode>(V)) {
      auto N = PHI->getNumOperands();
      if (N != 2)
        return false;
      return InVal1.match(PHI->getIncomingValue(0)) &&
             InVal2.match(PHI->getIncomingValue(1));
    }
    return false;
  }
};

template <typename InVal1Ty, typename InVal2Ty>
inline PHI_match<InVal1Ty, InVal2Ty> m_PHI(InVal1Ty InVal1, InVal2Ty InVal2) {
  return PHI_match<InVal1Ty, InVal2Ty>(InVal1, InVal2);
}

/// Class that matches pointer operation (GEP, Load, Store), capturing the
/// instruction and pointer.
template <typename Class, typename OpTy> struct bind_val_and_ptr_op_ty {
  Class *&VR;
  OpTy Op1;

  bind_val_and_ptr_op_ty(Class *&V, OpTy Op1) : VR(V), Op1(Op1) {}

  template <typename ITy> bool match(ITy *V) {
    auto *CV = dyn_cast<Class>(V);
    if (CV && Op1.match(CV->getPointerOperand())) {
      VR = CV;
      return true;
    }
    return false;
  }
};

/// Match a load, capturing the instruction and pointer if we match.
template <typename OpTy>
inline bind_val_and_ptr_op_ty<LoadInst, OpTy> m_LoadAndPtrOp(LoadInst *&Load,
                                                             OpTy Op) {
  return bind_val_and_ptr_op_ty<LoadInst, OpTy>(Load, Op);
}

/// This helper class is used to match GEP instruction.
/// Matches GetElementPointer instruction with pointer and first index, if
/// NumIndexes == 1, and both last and second last indexes, otherwise.
template <typename PtrTy, typename Idx1Ty,
          typename Idx2Ty = PatternMatch::is_zero>
struct GEP_match {
  GetElementPtrInst *&GEP;
  PtrTy Ptr;
  Idx1Ty Idx1;
  Idx2Ty Idx2;
  bool HasIdx2;

  GEP_match(GetElementPtrInst *&GEP, const PtrTy &Ptr, const Idx1Ty &Idx1)
      : GEP(GEP), Ptr(Ptr), Idx1(Idx1), Idx2(m_Zero()), HasIdx2(false) {}
  GEP_match(GetElementPtrInst *&GEP, const PtrTy &Ptr, const Idx1Ty &Idx1,
            const Idx2Ty &Idx2)
      : GEP(GEP), Ptr(Ptr), Idx1(Idx1), Idx2(Idx2), HasIdx2(true) {}

  template <typename OpTy> bool match(OpTy *V) {
    auto Matched = false;
    if (auto *I = dyn_cast<GetElementPtrInst>(V)) {
      GEP = I;
      auto N = I->getNumOperands();
      if (N > 2)
        Matched = Ptr.match(I->getOperand(0)) &&
                  Idx1.match(I->getOperand(N - 2)) &&
                  Idx2.match(I->getOperand(N - 1));
      else
        Matched = Ptr.match(I->getOperand(0)) && Idx1.match(I->getOperand(1));
    }
    return Matched;
  }
};

/// Matches GetElementPtrInst's PointerOperand and first Index value.
template <typename PtrTy, typename IdxTy>
inline GEP_match<PtrTy, IdxTy> m_GEP(GetElementPtrInst *&GEP, PtrTy Ptr,
                                     IdxTy Idx) {
  return GEP_match<PtrTy, IdxTy>(GEP, Ptr, Idx);
}

/// Matches GetElementPtrInst's PointerOperand and both last and second last
/// Index values.
template <typename PtrTy, typename Idx1Ty, typename Idx2Ty>
inline GEP_match<PtrTy, Idx1Ty, Idx2Ty>
m_GEP(GetElementPtrInst *&GEP, PtrTy Ptr, Idx1Ty Idx1, Idx2Ty Idx2) {
  return GEP_match<PtrTy, Idx1Ty, Idx2Ty>(GEP, Ptr, Idx1, Idx2);
}

// Base classs for m_OneOf matcher.
template <typename... List> struct match_one_of {
  // Empty instance should never be used.
  match_one_of() = delete;
};

// Empty list is never a match.
template <> struct match_one_of<> {
  template <typename ITy> bool match(ITy *V) { return false; }
};

// Matches either the head or tail of variadic OneOf argument list.
template <typename Head, typename... List> struct match_one_of<Head, List...> {
  Head Op;
  match_one_of<List...> Next;

  match_one_of(Head Op, List... Next) : Op(Op), Next(Next...) {}

  template <typename ITy> bool match(ITy *V) {
    return Op.match(V) || Next.match(V);
  }
};

/// This helper class is used to or-combine a list of matchers.
/// Matches one of the patterns in a list.
template <typename... PatternList>
inline match_one_of<PatternList...> m_OneOf(PatternList... Patterns) {
  return match_one_of<PatternList...>(Patterns...);
}

/// This helper class implements the same behavior as m_CombineOr but also
/// accepts a list of Value* that must be reset (set to nullptr) in case Op1
/// does not match. This behabior is usefull for matchers that bind an optional
/// value.
template <typename LHSTy, typename RHSTy, typename... ValuesTy>
struct CombineOrWithReset_match {
  LHSTy L;
  RHSTy R;
  SmallVector<Value **, 4> Values;

  CombineOrWithReset_match(LHSTy L, RHSTy R, ValuesTy... Values)
      : L(L), R(R), Values({Values...}) {}

  template <typename ITy> bool match(ITy *V) {
    if (L.match(V))
      return true;
    for (Value **ResetV : Values)
      if (ResetV != nullptr)
        (*ResetV) = nullptr;
    return R.match(V);
  }
};

/// Matches Op1 or Op2, setting ValuesTy to nullptr if Op1 doesn't match
template <typename Op1Ty, typename Op2Ty, typename... ValuesTy>
inline CombineOrWithReset_match<Op1Ty, Op2Ty, ValuesTy...>
m_CombineOrWithReset(const Op1Ty &Op1, const Op2Ty &Op2, ValuesTy... Values) {
  return CombineOrWithReset_match<Op1Ty, Op2Ty, ValuesTy...>(Op1, Op2,
                                                             Values...);
}

} // End namespace PatternMatch
} // End namespace llvm

namespace {
// A helper function that returns a matcher of a (optionally scaled)
// floating-point value. The return matcher matches expression in the form of
// alpha * X  or Y, where * is FMul instruction.
// TODO: add support for types other than floating-point
template <typename OpTy>
static inline auto scaledValueOrValue(OpTy V, Value *&Scalar) {
  return m_CombineOrWithReset(m_c_FMul(m_Value(Scalar), V), V, &Scalar);
}

// A helper function that returns a matcher of a linear function of two PHI
// instructions in the form of PHI1 * LD + PHI2, where LD is the angular
// coefficient of the linear function. The returned matcher captures both
// matched PHI instructions (PHI1 & PHI2) and the angular coefficient value
// (LD).
static inline auto linearFunctionOfPHI(PHINode *&PHI1, PHINode *&PHI2,
                                       Value *&LD) {
  return m_CombineOr(
      m_PHI(PHI1),
      m_c_Add(m_CombineOr(m_c_Mul(m_PHI(PHI1),
                                  m_OneOf(m_SExt(m_Value(LD)),
                                          m_ZExt(m_Value(LD)), m_Value(LD))),
                          m_Shl(m_PHI(PHI1), m_Value(LD))),
              m_PHI(PHI2)));
}

// A helper function that returns a matcher of an integer product between a PHI
// and a Value. The returned matcher captures both the PHINode (PHI) and the
// Value (LD). The matcher matches expresions of the form PHI * LD, where PHI
// is a PHINode and LD is a Value (optionally sign- or zero-extended).
static inline auto matchPHITimesLD(PHINode *&PHI, Value *&LD) {
  return m_CombineOr(
      m_c_Mul(m_PHI(PHI),
              m_OneOf(m_SExt(m_Value(LD)), m_ZExt(m_Value(LD)), m_Value(LD))),
      m_Shl(m_PHI(PHI), m_Value(LD)));
}

// A helper function that returns a matcher of a GEP instruction used to
// compute the effective address of a flat-array (1D) or 2D-array. The returned
// matcher captures the PointerOperand (Op), both PHINode instructions used to
// compute the effective address (PHI1 & PHI2) and, if used, the leading
// dimension value of the array (LD). The returned matcher accounts for
// different pattern of IR that are generated when accessing flat or 2D
// arrays.
static inline auto
match1Dor2DPtrOpAndInductionVariables(GetElementPtrInst *&GEPInst, Value *&Op,
                                      PHINode *&PHI1, PHINode *&PHI2,
                                      Value *&LD) {
  // TODO: Make dummy GEPs optional
  GetElementPtrInst *DummyGEP;
  return m_OneOf(
      m_GEP(DummyGEP, m_Load(m_GEP(GEPInst, m_Value(Op), m_PHI(PHI2))),
            m_PHI(PHI1)),
      m_GEP(DummyGEP, m_GEP(GEPInst, m_Value(Op), matchPHITimesLD(PHI1, LD)),
            m_PHI(PHI2)),
      m_GEP(DummyGEP, m_GEP(GEPInst, m_Value(Op), m_PHI(PHI2)),
            matchPHITimesLD(PHI1, LD)),
      m_GEP(GEPInst, m_Value(Op), m_PHI(PHI1), m_PHI(PHI2)),
      m_GEP(GEPInst, m_Value(Op), linearFunctionOfPHI(PHI1, PHI2, LD)));
}

// A helper function that returns a matcher of a load to flat or 2D array. The
// returned matcher uses the match1Dor2DPtrOpAndInductionVariables helper
// function's return as a sub-matcher.
static inline auto match1Dor2DLoadAndIndices(GetElementPtrInst *&GEPInst,
                                             Value *&PtrOp, PHINode *&Idx1,
                                             PHINode *&Idx2, Value *&LD) {
  return m_Load(
      match1Dor2DPtrOpAndInductionVariables(GEPInst, PtrOp, Idx1, Idx2, LD));
}

template <typename MulLHSTy, typename MulRHSTy>
static inline auto floatMultiplyWithScalar(MulLHSTy MulLHS, MulRHSTy MulRHS,
                                           Value *&Scalar) {
  return m_CombineOr(m_c_FMul(scaledValueOrValue(MulLHS, Scalar), MulRHS),
                     m_c_FMul(m_Value(Scalar), m_c_FMul(MulLHS, MulRHS)));
}

// A helper function that returns a matcher of a floating-point
// multiply-accumulate pattern involving two arrays. The returned matcher
// captures both multiplied arrays (MulLHS & MulRHS), the PHINode instructions
// used as indexes into each array (IdxA1/2 & IdxB1/2) and, if part of the
// indexing expression, the leading dimension of each array.
// TODO: add support for types other than floating-point
static inline auto
matchFMulFAddPattern(GetElementPtrInst *&GEPLHS, GetElementPtrInst *&GEPRHS,
                     Value *&MulLHS, Value *&MulRHS, PHINode *&IdxA1,
                     PHINode *&IdxA2, PHINode *&IdxB1, PHINode *&IdxB2,
                     Value *&LDA, Value *&LDB, Value *&Alpha) {
  auto LHS = match1Dor2DLoadAndIndices(GEPLHS, MulLHS, IdxA1, IdxA2, LDA);
  auto RHS = match1Dor2DLoadAndIndices(GEPRHS, MulRHS, IdxB1, IdxB2, LDB);
  return m_OneOf(
      m_c_FAdd(m_Value(), floatMultiplyWithScalar(LHS, RHS, Alpha)),
      m_Intrinsic<Intrinsic::fmuladd>(scaledValueOrValue(LHS, Alpha), RHS),
      m_Intrinsic<Intrinsic::fmuladd>(LHS, scaledValueOrValue(RHS, Alpha)));
}

// A helper function that returns a matcher of a store into a flat or 2D array
// C of the result of a reduction pattern matched by ReductionMatcher.
template <typename MatcherType>
static inline auto
matchStoreOfMatrixC(MatcherType ReductionMatcher, GetElementPtrInst *&GEPC,
                    Value *&C, PHINode *&IdxC1, PHINode *&IdxC2, Value *&LDC,
                    Value *&GEP, Value *&Alpha, Value *&Beta) {
  return m_Store(
      m_OneOf(scaledValueOrValue(ReductionMatcher, Alpha),
              scaledValueOrValue(m_PHI(m_Value(), ReductionMatcher), Beta),
              m_c_FAdd(scaledValueOrValue(m_Load(m_Value(GEP)), Beta),
                       scaledValueOrValue(
                           m_CombineOr(m_PHI(m_Zero(), ReductionMatcher),
                                       ReductionMatcher),
                           Alpha))),
      match1Dor2DPtrOpAndInductionVariables(GEPC, C, IdxC1, IdxC2, LDC));
}

// A helper function that returns a matcher of a store into a flat or 2D array
// C of the result of a reduction pattern matched by ReductionMatcher.
static inline auto matchSYR2KStore(const Value *C, Value *&AddLHS,
                                   Value *&AddRHS) {
  return m_CombineOr(
      m_Store(m_c_FAdd(m_PHI(m_Value(), m_Value()),
                       m_c_FAdd(m_Value(AddLHS), m_Value(AddRHS))),
              m_Specific(C)),
      m_Store(m_c_FAdd(m_Load(m_Specific(C)),
                       m_c_FAdd(m_Value(AddLHS), m_Value(AddRHS))),
              m_Specific(C)));
}

// A helper function that returns the outermost PHINode (induction variable)
// associated with V. The outermost induction variable has two incomming
// values, a initialization value (ConstantInt) and a post-increment value
// (AddInst). A chain of PHINodes is the IR pattern produced when compiling
// tiled loop nests.
static inline PHINode *extractOutermostPHI(PHINode *const &V) {
  if (!isa<PHINode>(V))
    return nullptr;

  SmallSetVector<const PHINode *, 8> WorkQueue;
  WorkQueue.insert(V);

  while (!WorkQueue.empty()) {
    const auto *PHI = WorkQueue.front();
    WorkQueue.remove(PHI);

    if (match(
            PHI,
            m_OneOf(m_PHI(m_c_Add(m_Specific(PHI), m_Value()), m_ConstantInt()),
                    m_PHI(m_ConstantInt(), m_c_Add(m_Specific(PHI), m_Value())),
                    m_PHI(m_ConstantInt(), m_ConstantInt()))))
      return const_cast<PHINode *>(PHI);

    for (const Use &Op : PHI->incoming_values())
      if (auto *InPHI = dyn_cast_or_null<PHINode>(&Op))
        WorkQueue.insert(InPHI);
  }
  return nullptr;
}

// A helper function that the inserts in Loops list the innermost loop nested
// in L, or L itself if L does not have sub-loops.
static void collecInnermostLoops(const Loop *L,
                                 SmallSetVector<const Loop *, 8> &Loops) {
  SmallSetVector<const Loop *, 8> WorkQueue;

  if (L->getSubLoops().size() == 0) {
    Loops.insert(L);
    return;
  }

  for (const auto *SL : L->getSubLoops())
    WorkQueue.insert(SL);

  while (!WorkQueue.empty()) {
    const auto *SL = WorkQueue.front();
    WorkQueue.remove(SL);

    bool HasSubLoop = false;
    for (const auto *SSL : SL->getSubLoops()) {
      HasSubLoop = true;
      WorkQueue.insert(SSL);
    }
    if (!HasSubLoop)
      Loops.insert(SL);
  }
}

// A helper function that collects into Loops all loops in a function that are
// nested at level 3 or deeper
static void
collectLoopsWithDepthThreeOrDeeper(LoopInfo &LI,
                                   SmallSetVector<const Loop *, 8> &Loops) {
  for (auto *L : LI.getLoopsInPreorder())
    if (L->getLoopDepth() > 2 && Loops.count(L) == 0)
      collecInnermostLoops(L, Loops);
}

// A helper function that returns true iff. either the first or the second
// operand of PHINode PHI matches the pattern described by matcher. If PHI has
// more than two operand or none operands matched then this function returns
// false. When true is returned the incoming argument AltValue is set to which
// ever operand did *NOT* match the pattern described by matcher.
template <typename MatcherTy>
static bool match1stOr2ndPHIOperand(const PHINode *&PHI, MatcherTy Matcher,
                                    Value *&AltValue) {
  if (PHI->getNumOperands() == 2) {
    Value *Op0 = PHI->getOperand(0);
    Value *Op1 = PHI->getOperand(1);
    bool Matched = true;
    if (match(Op0, Matcher))
      AltValue = Op1;
    else if (match(Op1, Matcher))
      AltValue = Op0;
    else
      Matched = false;
    return Matched;
  }
  return false;
}

// A helper function that tries to find the upper bound (UBound) of a loop
// associated with the induction variable (IndVar). If the upper bound is found
// and is a constant, then this function returns true and sets the incoming
// argument UBound to whichever value the loop associated with IndVar has.
// Otherwise, this function returns false and sets UBound to nullptr.
// If the incoming PHINode is a Phi(true, false) as in tripcount == 2 loops,
// the bound is not matched.
static bool matchLoopUpperBound(LoopInfo &LI, PHINode *IndVar, Value *&UBound) {
  BasicBlock *Header = IndVar->getParent();
  Loop *L = LI.getLoopFor(Header);
  if (L == nullptr)
    return false;
  SmallVector<BasicBlock *, 4> LoopExitingBBs;
  L->getExitingBlocks(LoopExitingBBs);

  // Iterate over branch instructions
  for (auto *BB : LoopExitingBBs) {
    auto *Term = BB->getTerminator();
    if (auto *BR = dyn_cast<BranchInst>(Term)) {
      // Pick the comparison instruction for this loop header
      for (auto *SuccBB : BR->successors()) {
        if (SuccBB != Header)
          continue;
        // For loops with trip_count == 2, Combine redundant instructions
        // replaces the integer induction variable with a phi(true, false)
        if (auto *Phi = dyn_cast<PHINode>(BR->getCondition()))
          if (match(Phi, m_CombineOr(m_PHI(m_Zero(), m_One()),
                                     m_PHI(m_One(), m_Zero())))) {
            IRBuilder<> IR(&Header->getParent()->getEntryBlock());
            UBound = IR.getInt64(2);
            return true;
          }
        if (auto *CMP = dyn_cast<ICmpInst>(BR->getCondition())) {
          // Iterate over header phis, find the one that matches the upper
          // bound
          for (BasicBlock::const_iterator I = Header->begin(); isa<PHINode>(I);
               I++) {
            const auto *PHI = static_cast<const Value *>(&*I);
            CmpPredicate CmpPred;
            if (!match(CMP,
                       m_c_ICmp(
                           CmpPred,
                           m_CombineOr(m_Specific(PHI),
                                       m_c_Add(m_Specific(PHI), m_Value())),
                           m_OneOf(m_ZExt(m_Value(UBound)),
                                   m_SExt(m_Value(UBound)), m_Value(UBound)))))
              continue;
            if (isa<PHINode>(UBound)) {
              // GEMM loops are not triangular.
              UBound = nullptr;
              return false;
            }
            return true;
          }
        }
      }
    }
  }
  return false;
}

// A helper function that returns the outer loop associated with one of the
// induction variables I, J, and K.
static Loop *getOuterLoop(LoopInfo &LI, Value *const &I, Value *const &J,
                          Value *const &K) {
  auto *A = LI.getLoopFor(static_cast<const PHINode *>(I)->getParent());
  auto *B = LI.getLoopFor(static_cast<const PHINode *>(J)->getParent());
  auto *C = LI.getLoopFor(static_cast<const PHINode *>(K)->getParent());
  auto *L = A->getLoopDepth() < B->getLoopDepth() ? A : B;
  return L->getLoopDepth() < C->getLoopDepth() ? L : C;
}

// A helper function that detects which access order (Layout) each matrix (A,
// B, and C) is used based on the induction variables A1/2, B1/2, and C1/2. The
// matrices involved in a GEMM of the form C += alpha * A * B + beta * C can
// only be accessed in the following combinations of access order (RM =
// Row-Major and CM = Column-Major):
//
//      |  C |  A |  B |
//      | RM | RM | RM |
//      | CM | RM | RM |
//      | RM | CM | RM |
//      | CM | CM | RM |
//      | RM | RM | CM |
//      | CM | RM | CM |
//      | RM | CM | CM |
//      | CM | CM | CM |
//
// If the access order is not one listed above then this function returns
// false, since their respective accesses are not part of a GEMM. Otherwise,
// this function returns true and ALayout, BLayout, and CLayout accordingly.
static bool matchMatrixLayout(PHINode *&A1, PHINode *&A2, PHINode *&B1,
                              PHINode *&B2, PHINode *&C1, PHINode *&C2,
                              CBLAS_ORDER &ALayout, CBLAS_ORDER &BLayout,
                              CBLAS_ORDER &CLayout, Value *&I, Value *&J,
                              Value *&K, LoopInfo &LI) {
  if (A1 == nullptr || A2 == nullptr || B1 == nullptr || B2 == nullptr ||
      C1 == nullptr || C2 == nullptr)
    return false;
  bool Matched = true;
  A1 = extractOutermostPHI(A1);
  A2 = extractOutermostPHI(A2);
  B1 = extractOutermostPHI(B1);
  B2 = extractOutermostPHI(B2);
  C1 = extractOutermostPHI(C1);
  C2 = extractOutermostPHI(C2);
  PHINode *II = nullptr;
  PHINode *JJ = nullptr;
  PHINode *KK = nullptr;
  if (A1 == B1) {
    II = A2;
    JJ = B2;
    KK = A1;
    ALayout = CBLAS_ORDER::ColMajor;
    BLayout = CBLAS_ORDER::RowMajor;
    if (A2 == C1 && B2 == C2)
      CLayout = CBLAS_ORDER::RowMajor;
    else if (A2 == C2 && B2 == C1)
      CLayout = CBLAS_ORDER::ColMajor;
    else
      // Not GEMM
      Matched = false;
  } else if (A1 == B2) {
    II = A2;
    JJ = B1;
    KK = A1;
    ALayout = CBLAS_ORDER::ColMajor;
    BLayout = CBLAS_ORDER::ColMajor;
    if (A2 == C1 && B1 == C2)
      CLayout = CBLAS_ORDER::RowMajor;
    else if (A2 == C2 && B1 == C1)
      CLayout = CBLAS_ORDER::ColMajor;
    else {
      // Not GEMM
      Matched = false;
    }
  } else if (A2 == B1) {
    II = A1;
    JJ = B2;
    KK = A2;
    ALayout = CBLAS_ORDER::RowMajor;
    BLayout = CBLAS_ORDER::RowMajor;
    if (A1 == C1 && B2 == C2)
      CLayout = CBLAS_ORDER::RowMajor;
    else if (A1 == C2 && B2 == C1)
      CLayout = CBLAS_ORDER::ColMajor;
    else {
      // Not GEMM
      Matched = false;
    }
  } else if (A2 == B2) {
    II = A1;
    JJ = B1;
    KK = A2;
    ALayout = CBLAS_ORDER::RowMajor;
    BLayout = CBLAS_ORDER::ColMajor;
    if (A1 == C1 && B1 == C2)
      CLayout = CBLAS_ORDER::RowMajor;
    else if (A1 == C2 && B1 == C1)
      CLayout = CBLAS_ORDER::ColMajor;
    else {
      // Not GEMM
      Matched = false;
    }
  } else {
    // Not GEMM
    Matched = false;
  }
  if (Matched) {
    auto DepthI = LI.getLoopDepth(II->getParent());
    auto DepthJ = LI.getLoopDepth(JJ->getParent());
    auto DepthK = LI.getLoopDepth(KK->getParent());
    if (DepthI != DepthJ && DepthJ != DepthK && DepthI != DepthK) {
      I = II;
      J = JJ;
      K = KK;
    } else {
      // Not GEMM
      Matched = false;
    }
  }
  return Matched;
}

static bool matchSyr2kIndVarAndLayout(Value *const &PtrToA, Value *&PtrToA2,
                                      Value *const &PtrToB, Value *&PtrToB2,
                                      SmallVector<PHINode *, 4> &APHI,
                                      SmallVector<PHINode *, 4> &BPHI,
                                      CBLAS_ORDER ALayout,
                                      CBLAS_ORDER BLayout) {

  // If RHS matched B as A and A as B, then we swap the matched induction
  // variables and pointers.
  if (PtrToA == PtrToB2 && PtrToB == PtrToA2) {
    std::swap(APHI[2], BPHI[2]);
    std::swap(APHI[3], BPHI[3]);
    std::swap(PtrToA2, PtrToB2);
  } else if (PtrToA != PtrToA2 || PtrToB != PtrToB2) {
    // If RHS' pointers do not match LHS' pointers, then this is not a Syr2k.
    return false;
  }

  if (ALayout == BLayout) {
    if (APHI[0] == BPHI[3] && APHI[1] == BPHI[2] && BPHI[0] == APHI[3] &&
        BPHI[1] == APHI[2])
      return true;
  } else if (APHI[0] == BPHI[2] && APHI[1] == BPHI[3] && BPHI[0] == APHI[2] &&
             BPHI[1] == APHI[3])
    return true;

  return false;
}

// A helper function that returns true iff. SeedInst is a store instruction
// that belongs to a Matrix-Multiply pattern. Otherwise this function returns
// false. When this function returns true all incoming arguments (except
// SeedInst) are set to capture values in the IR that describe the
// Matrix-Multiply.
static bool matchGEMM(Instruction &SeedInst, Value *&IVarI, Value *&IVarJ,
                      Value *&IVarK, GetElementPtrInst *&GEPA,
                      Value *&BasePtrToA, GetElementPtrInst *&GEPB,
                      Value *&BasePtrToB, GetElementPtrInst *&GEPC,
                      Value *&BasePtrToC, Value *&LDA, Value *&LDB, Value *&LDC,
                      CBLAS_ORDER &ALayout, CBLAS_ORDER &BLayout,
                      CBLAS_ORDER &CLayout, LoopInfo &LI, Value *&Alpha,
                      Value *&Beta, bool &IsCReduced) {
  auto *SeedInstAsValue = static_cast<Value *>(&SeedInst);
  Value *Alpha1 = nullptr;
  PHINode *APHI1 = nullptr;
  PHINode *APHI2 = nullptr;
  PHINode *BPHI1 = nullptr;
  PHINode *BPHI2 = nullptr;
  PHINode *CPHI1 = nullptr;
  PHINode *CPHI2 = nullptr;
  const auto *LoadPtrOp = SeedInst.getOperand(1);
  Value *MatchedGEP = nullptr;

  // MatMul matcher
  auto ReductionMatcher =
      matchFMulFAddPattern(GEPA, GEPB, BasePtrToA, BasePtrToB, APHI1, APHI2,
                           BPHI1, BPHI2, LDA, LDB, Alpha);
  auto Matcher = matchStoreOfMatrixC(ReductionMatcher, GEPC, BasePtrToC, CPHI1,
                                     CPHI2, LDC, MatchedGEP, Alpha1, Beta);

  bool IsGEMM = false;
  if (match(SeedInstAsValue, Matcher) && BasePtrToA != BasePtrToC &&
      BasePtrToB != BasePtrToC &&
      // prevents the match of double scaling with alpha
      (Alpha == nullptr || Alpha1 == nullptr) &&
      // LoadPtrOP equals MatchedGEP when old values of C is part of reduction,
      // i.e., when we have expressions of the form:
      // C = alpha? * A * B + beta? * C
      (MatchedGEP == nullptr || MatchedGEP == LoadPtrOp) &&
      matchMatrixLayout(APHI1, APHI2, BPHI1, BPHI2, CPHI1, CPHI2, ALayout,
                        BLayout, CLayout, IVarI, IVarJ, IVarK, LI)) {
    // If Alpha is not match with the reduction matcher, use the matched value
    // from the store matcher.
    if (Alpha == nullptr)
      Alpha = Alpha1;
    IsCReduced = MatchedGEP != nullptr || Beta != nullptr;
    IsGEMM = true;
  }
  return IsGEMM;
}

static bool matchSYR2K(Instruction &SeedInst, Value *&IVarI, Value *&IVarJ,
                       Value *&IVarK, GetElementPtrInst *&GEPA,
                       Value *&BasePtrToA, GetElementPtrInst *&GEPB,
                       Value *&BasePtrToB, GetElementPtrInst *&GEPC,
                       Value *&BasePtrToC, Value *&LDA, Value *&LDB,
                       Value *&LDC, CBLAS_ORDER &ALayout, CBLAS_ORDER &BLayout,
                       CBLAS_ORDER &CLayout, Value *&Alpha, Value *&Beta,
                       LoopInfo &LI) {
  Value *SeedInstAsValue = static_cast<Value *>(&SeedInst);
  Value *AddLHS = nullptr;
  Value *AddRHS = nullptr;
  SmallVector<PHINode *, 4> APHI = {nullptr, nullptr, nullptr, nullptr};
  SmallVector<PHINode *, 4> BPHI = {nullptr, nullptr, nullptr, nullptr};
  PHINode *CPHI1 = nullptr;
  PHINode *CPHI2 = nullptr;
  GetElementPtrInst *GEPA2 = nullptr;
  Value *BasePtrToA2 = nullptr;
  GetElementPtrInst *GEPB2 = nullptr;
  Value *BasePtrToB2 = nullptr;
  Value *LDA2 = nullptr;
  Value *LDB2 = nullptr;
  Value *Alpha1 = nullptr;
  Value *StoreToC = SeedInst.getOperand(1);

  auto StoreMatcher = matchSYR2KStore(StoreToC, AddLHS, AddRHS);

  auto AddLHSMatcher = floatMultiplyWithScalar(
      match1Dor2DLoadAndIndices(GEPA, BasePtrToA, APHI[0], APHI[1], LDA),
      match1Dor2DLoadAndIndices(GEPB, BasePtrToB, BPHI[0], BPHI[1], LDB),
      Alpha);

  auto AddRHSMatcher = floatMultiplyWithScalar(
      match1Dor2DLoadAndIndices(GEPA2, BasePtrToA2, APHI[2], APHI[3], LDA2),
      match1Dor2DLoadAndIndices(GEPB2, BasePtrToB2, BPHI[2], BPHI[3], LDB2),
      Alpha1);

  bool IsSYR2K = false;
  if (match(SeedInstAsValue, StoreMatcher) &&
      match(StoreToC, match1Dor2DPtrOpAndInductionVariables(
                          GEPC, BasePtrToC, CPHI1, CPHI2, LDC)) &&
      match(AddLHS, AddLHSMatcher) &&
      matchMatrixLayout(APHI[0], APHI[1], BPHI[0], BPHI[1], CPHI1, CPHI2,
                        ALayout, BLayout, CLayout, IVarI, IVarJ, IVarK, LI) &&
      match(AddRHS, AddRHSMatcher) &&
      matchSyr2kIndVarAndLayout(BasePtrToA, BasePtrToA2, BasePtrToB,
                                BasePtrToB2, APHI, BPHI, ALayout, BLayout))
    IsSYR2K = true;
  return IsSYR2K;
}

// A Helper function to get leading dimension value. In case V is the RHS of a
// Shl instruction, V is set to 1 << V. Otherwise V is unchanged.
static void setLeadingDimensionValue(BasicBlock &FunEntryBB, Loop &L,
                                     Value *const &IndVarA,
                                     Value *const &IndVarB, Value *&V) {
  for (auto *User : V->users()) {
    auto *Inst = dyn_cast_or_null<Instruction>(User);
    if (Inst != nullptr && L.contains(Inst->getParent()) &&
        match(User, m_Shl(m_CombineOr(m_Specific(IndVarA), m_Specific(IndVarB)),
                          m_Specific(V)))) {
      // Matched V is the RHS of a Shl instruction,
      // then actual V = 1 << matched(V)
      IRBuilder<> IR(&FunEntryBB);
      V = IR.CreateShl(IR.getInt64(1), V);
      break;
    }
  }
}

// A helper function that collects iniatialization stores to matrix C. This
// function adds to \p Stores all the store instructions that store the value 0
// to matrix C, which has base address \p C, in the effective address computed
// with \p LDC, IVarI, and IVarJ that are within loop \p L.
static void collectOtherKernelStoresToC(
    const GetElementPtrInst *GEPC, const Value *C, const Value *LDC,
    const Value *IVarI, const Value *IVarJ, const Value *Alpha,
    const Value *Beta, const Loop *L, const Instruction &ReductionStore,
    DominatorTree &DT, SmallSetVector<const Value *, 2> &Stores,
    bool &IsCReduced) {
  const auto *ReductionBB = ReductionStore.getParent();
  for (auto *BB : L->getBlocks()) {
    for (auto Inst = BB->begin(); Inst != BB->end(); Inst++) {
      if (isa<StoreInst>(Inst)) {
        auto *InstAsValue = static_cast<Value *>(&*Inst);
        Value *C1 = nullptr;
        Value *LDC1 = nullptr;
        PHINode *PHI1 = nullptr;
        PHINode *PHI2 = nullptr;
        Value *Alpha1 = nullptr;
        Value *Beta1 = nullptr;
        GetElementPtrInst *GEPC1 = nullptr;
        Value *GEP = Inst->getOperand(1);
        auto StoreMatcher = m_Store(
            m_OneOf(m_Constant(), scaledValueOrValue(m_Zero(), Alpha1),
                    scaledValueOrValue(m_Load(m_Specific(GEP)), Beta1),
                    m_c_FAdd(scaledValueOrValue(m_Load(m_Specific(GEP)), Beta1),
                             scaledValueOrValue(m_Zero(), Alpha1))),
            match1Dor2DPtrOpAndInductionVariables(GEPC1, C1, PHI1, PHI2, LDC1));
        if (match(InstAsValue, StoreMatcher) && GEPC == GEPC1 && C == C1 &&
            (LDC1 == nullptr || LDC1 == LDC) &&
            ((Alpha == nullptr || Alpha1 == nullptr) || Alpha == Alpha1) &&
            ((Beta == nullptr || Beta1 == nullptr) || Beta == Beta1) &&
            (IVarI == extractOutermostPHI(PHI1) ||
             IVarI == extractOutermostPHI(PHI2) ||
             IVarJ == extractOutermostPHI(PHI1) ||
             IVarJ == extractOutermostPHI(PHI2)) &&
            DT.dominates(BB, ReductionBB) && BB != ReductionBB) {
          if (Beta1 != nullptr)
            IsCReduced = true;
          Stores.insert(InstAsValue);
        }
      }
    }
  }
}

static Type *getMatrixElementType(GetElementPtrInst *GEP, bool &IsDoublePtr) {
  auto *GEPUser = GEP->getUniqueUndroppableUser();
  assert(GEPUser != nullptr &&
         "Matrix GetElementPtrInst must have a unique user!");
  assert((isa<LoadInst>(GEPUser) || isa<GetElementPtrInst>(GEPUser)) &&
         "GetElementPtrInst user must be a LoadInst!");
  if (const auto Load = dyn_cast<LoadInst>(GEPUser)) {
      auto *ElType = getLoadStoreType(Load);
      if (!ElType->isPointerTy()) {
        IsDoublePtr = false;
        return ElType;
      }
      auto *MaybeGEP =
          dyn_cast_or_null<GetElementPtrInst>(Load->getUniqueUndroppableUser());
      if (!MaybeGEP)
        return nullptr;
      auto *MaybeLoad =
          dyn_cast_or_null<LoadInst>(MaybeGEP->getUniqueUndroppableUser());
      if (!MaybeLoad)
        return nullptr;
      IsDoublePtr = true;
      return getLoadStoreType(MaybeLoad);
  } else { // GetElementPtrInst
    IsDoublePtr = false;
    const auto GEP2 = dyn_cast<GetElementPtrInst>(GEPUser);
    return GEP2->getSourceElementType();
  }
}

static Type *getCMatrixElementType(StoreInst *Store, bool &IsDoublePtr) {
  auto *GEP = dyn_cast_or_null<GetElementPtrInst>(getPointerOperand(Store));
  assert(GEP != nullptr &&
         "StoreInst pointer operand must be GetElementPtrInst!");
  IsDoublePtr = false;
  if (isa<LoadInst>(GEP->getPointerOperand()))
    IsDoublePtr = true;
  return getLoadStoreType(Store);
}

} // namespace

namespace KernelFaRer {

KernelMatcher::Result KernelMatcher::run(Function &F, LoopInfo &LI,
                                     DominatorTree &DT, OptimizationRemarkEmitter &ORE) {
  auto ListOfKernels =
      std::make_unique<SmallVector<std::unique_ptr<Kernel>, 4>>();
  SmallSetVector<const Loop *, 8> LoopsToProcess;
  collectLoopsWithDepthThreeOrDeeper(LI, LoopsToProcess);
  for (const auto *L : LoopsToProcess) {
    for (auto *BB : L->getParentLoop()->getBlocks()) {
      for (auto Inst = BB->begin(); Inst != BB->end(); Inst++) {
        if (!isa<StoreInst>(Inst))
          continue;
        GetElementPtrInst *GEPA = nullptr;
        Value *BasePtrToA = nullptr;
        GetElementPtrInst *GEPB = nullptr;
        Value *BasePtrToB = nullptr;
        GetElementPtrInst *GEPC = nullptr;
        Value *BasePtrToC = nullptr;
        Value *IVarI = nullptr;
        Value *IVarJ = nullptr;
        Value *IVarK = nullptr;
        Value *LDA = nullptr;
        Value *LDB = nullptr;
        Value *LDC = nullptr;
        Value *M = nullptr;
        Value *N = nullptr;
        Value *K = nullptr;
        Value *Alpha = nullptr;
        Value *Beta = nullptr;
        CBLAS_ORDER ALayout;
        CBLAS_ORDER BLayout;
        CBLAS_ORDER CLayout;
        bool IsCReduced = false;
        SmallSetVector<const llvm::Value *, 2> Stores;

        Kernel::KernelType KT = Kernel::KernelType::UNKNOWN_KERNEL;

        // Check that the loops for this store intstruction match the
        // Syr2k pattern
        if (matchSYR2K(*Inst, IVarI, IVarJ, IVarK, GEPA, BasePtrToA, GEPB,
                       BasePtrToB, GEPC, BasePtrToC, LDA, LDB, LDC, ALayout,
                       BLayout, CLayout, Alpha, Beta, LI) &&
            matchLoopUpperBound(LI, static_cast<PHINode *>(IVarJ), N) &&
            matchLoopUpperBound(LI, static_cast<PHINode *>(IVarK), K)) {
          KT = Kernel::KernelType::SYR2K_KERNEL;
          // Check that the loops for this store intstruction match the
          // Matrix-Multiply pattern
        } else if (matchGEMM(*Inst, IVarI, IVarJ, IVarK, GEPA, BasePtrToA, GEPB,
                             BasePtrToB, GEPC, BasePtrToC, LDA, LDB, LDC,
                             ALayout, BLayout, CLayout, LI, Alpha, Beta,
                             IsCReduced) &&
                   matchLoopUpperBound(LI, static_cast<PHINode *>(IVarI), M) &&
                   matchLoopUpperBound(LI, static_cast<PHINode *>(IVarJ), N) &&
                   matchLoopUpperBound(LI, static_cast<PHINode *>(IVarK), K)) {
          KT = Kernel::KernelType::GEMM_KERNEL;
        } else
          continue;

        Loop *OuterLoop = getOuterLoop(LI, IVarI, IVarJ, IVarK);
        // Verify that we only have one block we're exiting from.
        if (OuterLoop->getExitingBlock() == nullptr) {
          auto Header = OuterLoop->getHeader();
          auto DL = OuterLoop->getStartLoc();
          ORE.emit(
              OptimizationRemarkMissed(PASS_NAME, "MatcherRemark", DL, Header)
              << "Loop has multiple exiting blocks.");
          continue;
        }

        // Verify that we only have one exit block to go to. We won't have
        // any way to determine how to get to multiple exits.
        if (OuterLoop->getExitBlock() == nullptr) {
          auto Header = OuterLoop->getHeader();
          auto DL = OuterLoop->getStartLoc();
          ORE.emit(
              OptimizationRemarkMissed(PASS_NAME, "MatcherRemark", DL, Header)
              << "Loop has multiple exit blocks.");
          continue;
        }

        // Keep matched LDC for easier matching of initalization stores to
        // matrix C
        Value *MatchedLDC = LDC;

        Stores.insert(&*Inst);
        collectOtherKernelStoresToC(GEPC, BasePtrToC, MatchedLDC, IVarI, IVarJ,
                                    Alpha, Beta, OuterLoop, *Inst, DT, Stores,
                                    IsCReduced);

        bool IsADoublePtr = false;
        bool IsBDoublePtr = false;
        bool IsCDoublePtr = false;
        assert(GEPA && "No GetElementPtrInst match for matrix A!");
        Type *AElType = getMatrixElementType(GEPA, IsADoublePtr);
        assert(AElType && "Could not determine matrix A element type!");
        assert(GEPB && "No GetElementPtrInst match for matrix B!");
        Type *BElType = getMatrixElementType(GEPB, IsBDoublePtr);
        assert(BElType && "Could not determine matrix B element type!");
        assert(GEPC && "No GetElementPtrInst match for matrix C!");
        Type *CElType = getCMatrixElementType(static_cast<StoreInst *>(&*Inst),
                                              IsCDoublePtr);
        assert(CElType && "Could not determine matrix C element type!");
        if (KT == Kernel::KernelType::SYR2K_KERNEL) {
          // Note that LD* is determined first by the overall storage order
          // then whether or not the matrix has been transposed.
          if (LDA == nullptr)
            LDA = ALayout == KernelFaRer::RowMajor ? K : N;

          // TODO: C = A * B_T in SYR2K so matchMatrixLayout yields layout of
          // B_T
          if (LDB == nullptr)
            LDB = BLayout == KernelFaRer::RowMajor ? N : K;

          if (LDC == nullptr)
            LDC = N;

          // Matrices constructed from matched values and deduced layouts.
          Matrix MatrixA(*AElType, *BasePtrToA, ALayout, IsADoublePtr, *LDA, *N,
                         *K, *IVarI, *IVarK);
          Matrix MatrixB(*BElType, *BasePtrToB, BLayout, IsBDoublePtr, *LDB, *N,
                         *K, *IVarI, *IVarK);
          Matrix MatrixC(*CElType, *BasePtrToC, CLayout, IsCDoublePtr, *LDC, *N,
                         *N, *IVarI, *IVarJ);
          ListOfKernels->push_back(
              std::make_unique<SYR2K>(*OuterLoop, *Inst, MatrixA, MatrixB,
                                      MatrixC, Stores, Alpha, Beta));
        } else {
          // Note that LD* is determined first by the overall storage order
          // then whether or not the matrix has been transposed.
          if (LDA == nullptr) {
            if (CLayout == KernelFaRer::RowMajor)
              LDA = ALayout == CLayout ? K : M;
            else
              LDA = ALayout == CLayout ? M : K;
          } else {
            setLeadingDimensionValue(F.getEntryBlock(), *OuterLoop, IVarI,
                                     IVarK, LDA);
          }

          if (LDB == nullptr) {
            if (CLayout == KernelFaRer::RowMajor)
              LDB = BLayout == CLayout ? N : K;
            else
              LDB = BLayout == CLayout ? K : N;
          } else {
            setLeadingDimensionValue(F.getEntryBlock(), *OuterLoop, IVarK,
                                     IVarJ, LDB);
          }

          if (LDC == nullptr) {
            if (CLayout == KernelFaRer::RowMajor)
              LDC = N;
            else
              LDC = M;
          } else {
            setLeadingDimensionValue(F.getEntryBlock(), *OuterLoop, IVarI,
                                     IVarJ, LDC);
          }

          // Matrices constructed from matched values and deduced layouts.
          Matrix MatrixA(*AElType, *BasePtrToA, ALayout, IsADoublePtr, *LDA, *M,
                         *K, *IVarI, *IVarK);
          Matrix MatrixB(*BElType, *BasePtrToB, BLayout, IsBDoublePtr, *LDB, *K,
                         *N, *IVarK, *IVarJ);
          Matrix MatrixC(*CElType, *BasePtrToC, CLayout, IsCDoublePtr, *LDC, *M,
                         *N, *IVarI, *IVarJ);
          ListOfKernels->push_back(
              std::make_unique<GEMM>(*OuterLoop, *Inst, MatrixA, MatrixB,
                                     MatrixC, Stores, IsCReduced, Alpha, Beta));
        }
      }
    }
  }
  return ListOfKernels;
}

} // end of namespace KernelFaRer
