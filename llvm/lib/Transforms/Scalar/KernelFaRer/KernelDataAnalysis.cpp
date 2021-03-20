//===- GEMMDataAnalysis.cpp - Matrix-Multiply Analysis Pass -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analysis that returns true if a loop nest associated with Kernel can be
// deleted and replaced by a call to CBLAS or EIGEN.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/KernelFaRer.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "kernel-analysis"

using namespace llvm;

namespace KernelFaRer {

/// Analysis that returns true if a loop nest associated with Kernel can be
/// deleted and replaced by a call to CBLAS or EIGEN.
///
/// A loop nest L associated with \p Kernel cannot be replaced with
/// llvm.matrix.multiply.* instrinsics if any of the following conditions hold:
///   - L contains any store instructions that write to memory other than to
///   \p Ker's result matrix;
///   - L contains instructions with side-effects (e.g. I/O operations);
///   - L constains instructions that define values used outside of L.
bool KernelDataAnalysisPass::run(Kernel &Ker) {
  auto &L = Ker.getAssociatedLoop();

  for (const auto &BB : L.getBlocks())
    for (const auto &Inst : *BB) {
      if (Ker.isKernelStore(Inst) || Ker.isKernelValue(Inst))
        // Skip stores and values that belongs to Kernel
        continue;
      // Instructions with side-effects are *NOT* allowed.
      if (Inst.mayHaveSideEffects())
        return false;
      // Any instructions that write to memory other than to Kernel's result
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

} // end of namespace KernelFaRer

#undef DEBUG_TYPE
