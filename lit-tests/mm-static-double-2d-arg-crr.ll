; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --scrub-attributes
; RUN: opt -passes='gemm-replacer-pass,loop-deletion,dce,simplifycfg' --gemmfarer-replacement-mode=matrix-intrinsics -S < %s | FileCheck %s

define void @_Z11fixedSizeMMPA16_KdPA8_S_PA8_d(ptr %A, ptr %B, ptr %C) {
; CHECK-LABEL: @_Z11fixedSizeMMPA16_KdPA8_S_PA8_d(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call <1024 x double> @llvm.matrix.column.major.load.v1024f64.i64(ptr align 8 [[A:%.*]], i64 16, i1 false, i32 16, i32 64)
; CHECK-NEXT:    [[TMP1:%.*]] = call <512 x double> @llvm.matrix.column.major.load.v512f64.i64(ptr align 8 [[B:%.*]], i64 8, i1 false, i32 8, i32 64)
; CHECK-NEXT:    [[TMP2:%.*]] = call <512 x double> @llvm.matrix.transpose.v512f64(<512 x double> [[TMP1]], i32 8, i32 64)
; CHECK-NEXT:    [[TMP3:%.*]] = call <128 x double> @llvm.matrix.multiply.v128f64.v1024f64.v512f64(<1024 x double> [[TMP0]], <512 x double> [[TMP2]], i32 16, i32 64, i32 8)
; CHECK-NEXT:    [[TMP4:%.*]] = call <128 x double> @llvm.matrix.transpose.v128f64(<128 x double> [[TMP3]], i32 16, i32 8)
; CHECK-NEXT:    call void @llvm.matrix.column.major.store.v128f64.i64(<128 x double> [[TMP4]], ptr align 8 [[C:%.*]], i64 8, i1 false, i32 8, i32 16)
; CHECK-NEXT:    ret void
;
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %indvars.iv48 = phi i64 [ 0, %entry ], [ %indvars.iv.next49, %for.cond.cleanup3 ]
  br label %for.cond5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void

for.cond5.preheader:                              ; preds = %for.cond1.preheader, %for.cond.cleanup7
  %indvars.iv45 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next46, %for.cond.cleanup7 ]
  br label %for.body8

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %indvars.iv.next49 = add nuw nsw i64 %indvars.iv48, 1
  %exitcond50.not = icmp eq i64 %indvars.iv.next49, 16
  br i1 %exitcond50.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup7:                                ; preds = %for.body8
  %arrayidx18 = getelementptr inbounds [8 x double], ptr %C, i64 %indvars.iv48, i64 %indvars.iv45
  store double %add, ptr %arrayidx18, align 8
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond47.not = icmp eq i64 %indvars.iv.next46, 8
  br i1 %exitcond47.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:                                        ; preds = %for.cond5.preheader, %for.body8
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %c.041 = phi double [ 0.000000e+00, %for.cond5.preheader ], [ %add, %for.body8 ]
  %arrayidx10 = getelementptr inbounds [16 x double], ptr %A, i64 %indvars.iv, i64 %indvars.iv48
  %0 = load double, ptr %arrayidx10, align 8
  %arrayidx14 = getelementptr inbounds [8 x double], ptr %B, i64 %indvars.iv, i64 %indvars.iv45
  %1 = load double, ptr %arrayidx14, align 8
  %mul = fmul double %0, %1
  %add = fadd double %c.041, %mul
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8
}
