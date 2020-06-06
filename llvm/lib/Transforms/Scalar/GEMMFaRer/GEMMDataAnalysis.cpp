//===- GEMMDataAnalysis.cpp - Matrix-Multiply Analysis Pass -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analysis that returns true if a loop nest associated with a given GEMM
// (GeMM) can be deleted and replaced by a call to llvm.matrix.multiply.*.
//
//===----------------------------------------------------------------------===//
//
// TODO List:
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/GEMMFaRer.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "gemm-analysis"

using namespace llvm;

namespace GEMMFaRer {

/// Analysis that returns true if a loop nest associated with GeMM can be
/// deleted and replaced by a call to llvm.matrix.multiply.*.
///
/// A loop nest L associated with \p GeMM cannot be replaced with
/// llvm.matrix.multiply.* instrinsics if any of the following conditions hold:
///   - L contains any store instructions that write to memory other than to
///   \p GeMM's result matrix;
///   - L contains instructions with side-effects (e.g. I/O operations);
///   - L constains instructions that define values used outside of L.
bool GEMMDataAnalysisPass::run(GEMM &GeMM) {
  auto &L = GeMM.getAssociatedLoop();

  for (const auto &BB : L.getBlocks())
    for (const auto &Inst : *BB) {
      if (GeMM.isGemmStore(Inst) || GeMM.isGemmValue(Inst))
        // Skip stores and values that belongs to GeMM
        continue;
      // Instructions with side-effects are *NOT* allowed.
      if (Inst.mayHaveSideEffects())
        return false;
      // Any instructions that write to memory other than to GeMM's result
      // matrix are also *NOT* allowed.
      if (Inst.mayWriteToMemory())
        return false;
      // Instructions that neither produce side-effects nor write to memory need
      // to be further checked to see if their users define value used outside
      // of L. If so, L cannot be deleted because otherwise such values will not
      // be produced.
      for (const auto *User : Inst.users()) {
        const auto *UserAsInst = dyn_cast<Instruction>(User);
        if (!L.contains(UserAsInst->getParent()))
          return false;
      }
    }

  return true;
}

} // end of namespace GEMMFaRer

#undef DEBUG_TYPE
