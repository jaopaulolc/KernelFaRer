; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --scrub-attributes
; RUN: opt -passes='gemm-replacer-pass,loop-deletion,dce,simplifycfg' --gemmfarer-replacement-mode=matrix-intrinsics -S < %s | FileCheck %s

define void @_Z11fixedSizeMMPKfiS0_iPfi(ptr %A, i32 %lda, ptr %B, i32 %ldb, ptr %C, i32 %ldc) {
; CHECK-LABEL: @_Z11fixedSizeMMPKfiS0_iPfi(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = sext i32 [[LDA:%.*]] to i64
; CHECK-NEXT:    [[TMP1:%.*]] = sext i32 [[LDB:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = sext i32 [[LDC:%.*]] to i64
; CHECK-NEXT:    [[TMP3:%.*]] = call <1024 x float> @llvm.matrix.column.major.load.v1024f32.i64(ptr align 4 [[A:%.*]], i64 [[TMP0]], i1 false, i32 64, i32 16)
; CHECK-NEXT:    [[TMP4:%.*]] = call <1024 x float> @llvm.matrix.transpose.v1024f32(<1024 x float> [[TMP3]], i32 64, i32 16)
; CHECK-NEXT:    [[TMP5:%.*]] = call <512 x float> @llvm.matrix.column.major.load.v512f32.i64(ptr align 4 [[B:%.*]], i64 [[TMP1]], i1 false, i32 64, i32 8)
; CHECK-NEXT:    [[TMP6:%.*]] = call <128 x float> @llvm.matrix.multiply.v128f32.v1024f32.v512f32(<1024 x float> [[TMP4]], <512 x float> [[TMP5]], i32 16, i32 64, i32 8)
; CHECK-NEXT:    [[TMP7:%.*]] = call <128 x float> @llvm.matrix.transpose.v128f32(<128 x float> [[TMP6]], i32 16, i32 8)
; CHECK-NEXT:    call void @llvm.matrix.column.major.store.v128f32.i64(<128 x float> [[TMP7]], ptr align 4 [[C:%.*]], i64 [[TMP2]], i1 false, i32 8, i32 16)
; CHECK-NEXT:    ret void
;
entry:
  %0 = sext i32 %ldb to i64
  %1 = sext i32 %lda to i64
  %2 = sext i32 %ldc to i64
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %indvars.iv55 = phi i64 [ 0, %entry ], [ %indvars.iv.next56, %for.cond.cleanup3 ]
  %3 = mul nsw i64 %indvars.iv55, %1
  %4 = mul nsw i64 %indvars.iv55, %2
  br label %for.cond5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void

for.cond5.preheader:                              ; preds = %for.cond1.preheader, %for.cond.cleanup7
  %indvars.iv50 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next51, %for.cond.cleanup7 ]
  %5 = mul nsw i64 %indvars.iv50, %0
  br label %for.body8

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %indvars.iv.next56 = add nuw nsw i64 %indvars.iv55, 1
  %exitcond59.not = icmp eq i64 %indvars.iv.next56, 16
  br i1 %exitcond59.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup7:                                ; preds = %for.body8
  %6 = add nsw i64 %indvars.iv50, %4
  %arrayidx18 = getelementptr inbounds float, ptr %C, i64 %6
  store float %add14, ptr %arrayidx18, align 4
  %indvars.iv.next51 = add nuw nsw i64 %indvars.iv50, 1
  %exitcond54.not = icmp eq i64 %indvars.iv.next51, 8
  br i1 %exitcond54.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:                                        ; preds = %for.cond5.preheader, %for.body8
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %c.044 = phi float [ 0.000000e+00, %for.cond5.preheader ], [ %add14, %for.body8 ]
  %7 = add nsw i64 %indvars.iv, %3
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %7
  %8 = load float, ptr %arrayidx, align 4
  %9 = add nsw i64 %indvars.iv, %5
  %arrayidx12 = getelementptr inbounds float, ptr %B, i64 %9
  %10 = load float, ptr %arrayidx12, align 4
  %mul13 = fmul float %8, %10
  %add14 = fadd float %c.044, %mul13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8
}
