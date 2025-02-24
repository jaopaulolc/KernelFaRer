; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --scrub-attributes
; RUN: opt -passes='gemm-replacer-pass,loop-deletion,dce,simplifycfg' --gemmfarer-replacement-mode=matrix-intrinsics -S < %s | FileCheck %s

define void @_Z11fixedSizeMMPKdiS0_iPdi(ptr %A, i32 %lda, ptr %B, i32 %ldb, ptr %C, i32 %ldc) {
; CHECK-LABEL: @_Z11fixedSizeMMPKdiS0_iPdi(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = sext i32 [[LDA:%.*]] to i64
; CHECK-NEXT:    [[TMP1:%.*]] = sext i32 [[LDB:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = sext i32 [[LDC:%.*]] to i64
; CHECK-NEXT:    [[TMP3:%.*]] = call <1024 x double> @llvm.matrix.column.major.load.v1024f64.i64(ptr align 8 [[A:%.*]], i64 [[TMP0]], i1 false, i32 64, i32 16)
; CHECK-NEXT:    [[TMP4:%.*]] = call <1024 x double> @llvm.matrix.transpose.v1024f64(<1024 x double> [[TMP3]], i32 64, i32 16)
; CHECK-NEXT:    [[TMP5:%.*]] = call <512 x double> @llvm.matrix.column.major.load.v512f64.i64(ptr align 8 [[B:%.*]], i64 [[TMP1]], i1 false, i32 8, i32 64)
; CHECK-NEXT:    [[TMP6:%.*]] = call <512 x double> @llvm.matrix.transpose.v512f64(<512 x double> [[TMP5]], i32 8, i32 64)
; CHECK-NEXT:    [[TMP7:%.*]] = call <128 x double> @llvm.matrix.multiply.v128f64.v1024f64.v512f64(<1024 x double> [[TMP4]], <512 x double> [[TMP6]], i32 16, i32 64, i32 8)
; CHECK-NEXT:    [[TMP8:%.*]] = call <128 x double> @llvm.matrix.transpose.v128f64(<128 x double> [[TMP7]], i32 16, i32 8)
; CHECK-NEXT:    call void @llvm.matrix.column.major.store.v128f64.i64(<128 x double> [[TMP8]], ptr align 8 [[C:%.*]], i64 [[TMP2]], i1 false, i32 8, i32 16)
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
  %indvars.iv51 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next52, %for.cond.cleanup7 ]
  br label %for.body8

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %indvars.iv.next56 = add nuw nsw i64 %indvars.iv55, 1
  %exitcond59.not = icmp eq i64 %indvars.iv.next56, 16
  br i1 %exitcond59.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup7:                                ; preds = %for.body8
  %5 = add nsw i64 %indvars.iv51, %4
  %arrayidx18 = getelementptr inbounds double, ptr %C, i64 %5
  store double %add14, ptr %arrayidx18, align 8
  %indvars.iv.next52 = add nuw nsw i64 %indvars.iv51, 1
  %exitcond54.not = icmp eq i64 %indvars.iv.next52, 8
  br i1 %exitcond54.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:                                        ; preds = %for.cond5.preheader, %for.body8
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %c.044 = phi double [ 0.000000e+00, %for.cond5.preheader ], [ %add14, %for.body8 ]
  %6 = add nsw i64 %indvars.iv, %3
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %6
  %7 = load double, ptr %arrayidx, align 8
  %8 = mul nsw i64 %indvars.iv, %0
  %9 = add nsw i64 %8, %indvars.iv51
  %arrayidx12 = getelementptr inbounds double, ptr %B, i64 %9
  %10 = load double, ptr %arrayidx12, align 8
  %mul13 = fmul double %7, %10
  %add14 = fadd double %c.044, %mul13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8
}
