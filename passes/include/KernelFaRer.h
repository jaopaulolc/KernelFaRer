//===- KernelFaRer.h - Matrix-Multiply Replacer Pass -------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements an idiom recognizer that transforms matrix-multiply
// and syr2k loops into a call high-performance libraries or to
// llvm.matrix.multiply.* intrinsic.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_KERNELFARER_H
#define LLVM_TRANSFORMS_SCALAR_KERNELFARER_H

#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"

using namespace llvm;

namespace KernelFaRer {

/// Available replacement modes
///
/// EIGEN - matched GEMM is replaced with calls to eigen-runtime
/// CBLAS - matched Kernel is replaced with a call to cblas_X
/// MatrixIntrinsics - matched GEMM is replaced with calls to llvm.matrix.*
enum ReplacementMode : uint8_t {
  EIGEN,
  CBLAS,
  MatrixIntrinsics,
  UNKNOWN = 0xFF
};

// These constants are from
// https://github.com/xianyi/OpenBLAS/blob/develop/cblas.h#L54-L58
enum CBLAS_ORDER { RowMajor = 101, ColMajor = 102 };
enum CBLAS_TRANSPOSE {
  NoTrans = 111,
  Trans = 112,
  ConjTrans = 113,
  ConjNoTrans = 114
};
enum CBLAS_UPLO { Upper = 121, Lower = 122 };
enum CBLAS_DIAG { NonUnit = 131, Unit = 132 };
enum CBLAS_SIDE { Left = 141, Right = 142 };

/// Class that represents a matrix extracted from GEMM
class Matrix {
  Type &EltType;
  Value &BaseAddressPointer;   ///< pointer to the array of matrix elements
  CBLAS_ORDER Layout;          ///< layout of matrix elements in the array
  bool IsDoublePtr;            ///< true if matrix is double pointer of EltType
  Value &LeadingDimensionSize; ///< number of elements between two consecutive
                               ///< columns if ColMajor or consecutive rows if
                               ///< RowMajor
  Value &Rows;                 ///< number of Rows
  Value &Columns;              ///< number of Columns
  Value &RowIV;                ///< induction variable to access rows
  Value &ColumnIV;             ///< induction variable to access columns

public:
  Matrix(Type &EltType, Value &BaseAddressPointer, CBLAS_ORDER Layout,
         bool IsDoublePtr, Value &LeadingDimensionSize, Value &Rows,
         Value &Columns, Value &RowIV, Value &ColumnIV)
      : EltType(EltType), BaseAddressPointer(BaseAddressPointer),
        Layout(Layout), IsDoublePtr(IsDoublePtr),
        LeadingDimensionSize(LeadingDimensionSize), Rows(Rows),
        Columns(Columns), RowIV(RowIV), ColumnIV(ColumnIV) {}

  Type &getElementType() const { return EltType; }
  Type *getScalarElementType() const {
    Type *ElementType = &EltType;
    if (ElementType->isArrayTy()) {
      ElementType = ElementType->getArrayElementType();
      if (ElementType->isArrayTy())
        ElementType = ElementType->getArrayElementType();
    }
    assert(ElementType->isIntegerTy() || ElementType->isFloatingPointTy());
    return ElementType;
  }
  Value &getBaseAddressPointer() const { return BaseAddressPointer; }
  CBLAS_ORDER getLayout() const { return Layout; }
  bool isDoublePtr() const { return IsDoublePtr; }
  Value &getLeadingDimensionSize() const { return LeadingDimensionSize; }
  Value &getRows() const { return Rows; }
  Value &getColumns() const { return Columns; }
  Value &getRowIV() const { return RowIV; }
  Value &getColumnIV() const { return ColumnIV; }
}; // class Matrix

/// Abstract class that represents a kernel computed within a triple-nested
/// loop.
class Kernel {
public:
  enum KernelType { GEMM_KERNEL, SYR2K_KERNEL, UNKNOWN_KERNEL = 0xff };

protected:
  KernelType KernelID;
  Loop &L;
  Instruction &ReductionStore;
  Matrix MatrixA;
  Matrix MatrixB;
  Matrix MatrixC;
  Value *Alpha;
  Value *Beta;
  SmallSetVector<const Value *, 2> Stores;

public:
  Kernel(KernelType KernelID, Loop &L, Instruction &RS, Matrix &MatrixA,
         Matrix &MatrixB, Matrix &MatrixC,
         SmallSetVector<const Value *, 2> Stores, Value *Alpha = nullptr,
         Value *Beta = nullptr)
      : KernelID(KernelID), L(L), ReductionStore(RS),
        MatrixA(std::move(MatrixA)), MatrixB(std::move(MatrixB)),
        MatrixC(std::move(MatrixC)), Alpha(Alpha), Beta(Beta),
        Stores(std::move(Stores)) {}
  virtual ~Kernel() = default;
  Loop &getAssociatedLoop() const { return L; }
  Instruction &getReductionStore() const { return ReductionStore; }
  const Matrix &getMatrixA() const { return MatrixA; }
  const Matrix &getMatrixB() const { return MatrixB; }
  const Matrix &getMatrixC() const { return MatrixC; }
  Value *getAlpha() const { return Alpha; }
  Value *getBeta() const { return Beta; }
  KernelType getKernelID() const { return KernelID; }

  virtual bool isKernelStore(const Value &Store) const = 0;
  virtual bool isKernelValue(const Value &V) const = 0;
}; // class Kernel

/// Class that represents a triple-nested loop that computes a general
/// matrix-matrix multiply.
class GEMM : public Kernel {
  bool CIsReduced;

public:
  GEMM(Loop &L, Instruction &RS, Matrix &MatrixA, Matrix &MatrixB,
       Matrix &MatrixC, SmallSetVector<const Value *, 2> Stores,
       bool CIsReduced, Value *Alpha = nullptr, Value *Beta = nullptr)
      : Kernel(Kernel::GEMM_KERNEL, L, RS, MatrixA, MatrixB, MatrixC, Stores,
               Alpha, Beta),
        CIsReduced(CIsReduced) {}

  /// This method indicates if C is part of the reduction or not.
  bool IsCReduced() const { return CIsReduced; }

  /// This predicate method determines if \p Store is a store to GEMM's result
  /// matrix.
  ///
  /// \param Store a StoreInst to be checked if it a store to GEMM's result
  /// matrix.
  ///
  /// \returns true if \p Store store into GEMM's result matrix and false
  /// otherwise.
  bool isKernelStore(const Value &Store) const override {
    return Stores.count(&Store) != 0;
  }

  /// This predicate method determines if \p V is one of the values used to
  /// compute this GEMM.
  ///
  /// \param V a value to be checked if it is used to compute GEMM
  ///
  /// \returns true if \p V is used to compute GEMM and false otherwise.
  bool isKernelValue(const Value &V) const override {

    auto &M = MatrixA.getRows();
    auto &K = MatrixA.getColumns();
    auto &N = MatrixB.getColumns();

    auto &IndVarI = MatrixA.getRowIV();
    auto &IndVarK = MatrixA.getColumnIV();
    auto &IndVarJ = MatrixB.getColumnIV();

    auto &ABaseAddr = MatrixA.getBaseAddressPointer();
    auto &BBaseAddr = MatrixB.getBaseAddressPointer();
    auto &CBaseAddr = MatrixC.getBaseAddressPointer();

    auto &LDA = MatrixA.getLeadingDimensionSize();
    auto &LDB = MatrixB.getLeadingDimensionSize();
    auto &LDC = MatrixC.getLeadingDimensionSize();

    return (&V == Alpha || &V == Beta || &V == &IndVarI || &V == &IndVarJ ||
            &V == &IndVarK || &V == &M || &V == &N || &V == &K ||
            &V == &ABaseAddr || &V == &BBaseAddr || &V == &CBaseAddr ||
            &V == &LDA || &V == &LDB || &V == &LDC);
  }

  static inline bool classof(GEMM const *) { return true; }
  static inline bool classof(Kernel const *K) {
    return K->getKernelID() == Kernel::GEMM_KERNEL;
  }
}; // class GEMM

class SYR2K : public Kernel {
  Value *Uplo;

public:
  SYR2K(Loop &L, Instruction &RS, Matrix &MatrixA, Matrix &MatrixB,
        Matrix &MatrixC, SmallSetVector<const Value *, 2> Stores,
        Value *Alpha = nullptr, Value *Beta = nullptr, Value *Uplo = nullptr)
      : Kernel(Kernel::SYR2K_KERNEL, L, RS, MatrixA, MatrixB, MatrixC, Stores,
               Alpha, Beta),
        Uplo(Uplo) {}

  Value *getUplo() const { return Uplo; }

  bool isKernelStore(const Value &Store) const override {
    return Stores.count(&Store) != 0;
  }

  bool isKernelValue(const Value &V) const override {

    auto &N = MatrixA.getRows();
    auto &K = MatrixA.getColumns();

    auto &IndVarI = MatrixA.getRowIV();
    auto &IndVarK = MatrixA.getColumnIV();
    auto &IndVarJ = MatrixB.getColumnIV();

    return (&V == Alpha || &V == Beta || &V == &IndVarI || &V == &IndVarJ ||
            &V == &IndVarK || &V == &N || &V == &K ||
            &V == &MatrixA.getBaseAddressPointer() ||
            &V == &MatrixB.getBaseAddressPointer() ||
            &V == &MatrixC.getBaseAddressPointer() ||
            &V == &MatrixA.getLeadingDimensionSize() ||
            &V == &MatrixB.getLeadingDimensionSize() ||
            &V == &MatrixC.getLeadingDimensionSize());
  }

  static inline bool classof(SYR2K const *) { return true; }
  static inline bool classof(Kernel const *K) {
    return K->getKernelID() == Kernel::SYR2K_KERNEL;
  }
}; // class SYR2K

/// Performs Kernel Recognition Pass.
struct KernelMatcher {
public:
  using Result = std::unique_ptr<SmallVector<std::unique_ptr<Kernel>, 4>>;
  static Result run(Function &F, LoopInfo &LI, DominatorTree &DT, OptimizationRemarkEmitter &ORE);
};

/// Checks if Kernel can be replaced
struct DataAnalysisPass {
public:
  static bool run(Kernel &Ker);
};

} // end of namespace KernelFaRer

namespace llvm {
/// Performs Kernel Replacement Pass.
struct KernelReplacerPass : public PassInfoMixin<KernelReplacerPass> {
  friend PassInfoMixin<KernelReplacerPass>;

public:
  static PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // end of namespace llvm

#endif /* LLVM_TRANSFORMS_SCALAR_KERNELFARER_H */
