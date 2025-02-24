; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --scrub-attributes
; RUN: opt -passes='gemm-replacer-pass,loop-deletion,dce,simplifycfg,loop-deletion,dce,simplifycfg' --gemmfarer-replacement-mode=matrix-intrinsics -S < %s | FileCheck %s

define void @_Z14staticSizeGEMMPA64_KdS1_PA8_ddd(ptr %A, ptr %B, ptr %C, double %alpha, double %beta) {
; CHECK-LABEL: @_Z14staticSizeGEMMPA64_KdS1_PA8_ddd(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call <1024 x double> @llvm.matrix.column.major.load.v1024f64.i64(ptr align 8 [[A:%.*]], i64 64, i1 false, i32 64, i32 16)
; CHECK-NEXT:    [[TMP1:%.*]] = call <1024 x double> @llvm.matrix.transpose.v1024f64(<1024 x double> [[TMP0]], i32 64, i32 16)
; CHECK-NEXT:    [[TMP2:%.*]] = call <512 x double> @llvm.matrix.column.major.load.v512f64.i64(ptr align 8 [[B:%.*]], i64 64, i1 false, i32 64, i32 8)
; CHECK-NEXT:    [[TMP3:%.*]] = call <128 x double> @llvm.matrix.multiply.v128f64.v1024f64.v512f64(<1024 x double> [[TMP1]], <512 x double> [[TMP2]], i32 16, i32 64, i32 8)
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLATINSERT:%.*]] = insertelement <128 x double> poison, double [[ALPHA:%.*]], i32 0
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLAT:%.*]] = shufflevector <128 x double> [[SCALAR_SPLAT_SPLATINSERT]], <128 x double> poison, <128 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = fmul <128 x double> [[SCALAR_SPLAT_SPLAT]], [[TMP3]]
; CHECK-NEXT:    [[TMP5:%.*]] = call <128 x double> @llvm.matrix.column.major.load.v128f64.i64(ptr align 8 [[C:%.*]], i64 8, i1 false, i32 8, i32 16)
; CHECK-NEXT:    [[TMP6:%.*]] = call <128 x double> @llvm.matrix.transpose.v128f64(<128 x double> [[TMP5]], i32 8, i32 16)
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLATINSERT1:%.*]] = insertelement <128 x double> poison, double [[BETA:%.*]], i32 0
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLAT2:%.*]] = shufflevector <128 x double> [[SCALAR_SPLAT_SPLATINSERT1]], <128 x double> poison, <128 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = fmul <128 x double> [[SCALAR_SPLAT_SPLAT2]], [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = fadd <128 x double> [[TMP4]], [[TMP7]]
; CHECK-NEXT:    [[TMP9:%.*]] = call <128 x double> @llvm.matrix.transpose.v128f64(<128 x double> [[TMP8]], i32 16, i32 8)
; CHECK-NEXT:    call void @llvm.matrix.column.major.store.v128f64.i64(<128 x double> [[TMP9]], ptr align 8 [[C]], i64 8, i1 false, i32 8, i32 16)
; CHECK-NEXT:    ret void
;
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %indvars.iv58 = phi i64 [ 0, %entry ], [ %indvars.iv.next59, %for.cond.cleanup3 ]
  br label %for.cond5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void

for.cond5.preheader:                              ; preds = %for.cond1.preheader, %for.cond.cleanup7
  %indvars.iv55 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next56, %for.cond.cleanup7 ]
  br label %for.body8

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %indvars.iv.next59 = add nuw nsw i64 %indvars.iv58, 1
  %exitcond60.not = icmp eq i64 %indvars.iv.next59, 16
  br i1 %exitcond60.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup7:                                ; preds = %for.body8
  %mul15 = fmul double %add, %alpha
  %arrayidx19 = getelementptr inbounds [8 x double], ptr %C, i64 %indvars.iv58, i64 %indvars.iv55
  %0 = load double, ptr %arrayidx19, align 8
  %mul20 = fmul double %0, %beta
  %add21 = fadd double %mul15, %mul20
  store double %add21, ptr %arrayidx19, align 8
  %indvars.iv.next56 = add nuw nsw i64 %indvars.iv55, 1
  %exitcond57.not = icmp eq i64 %indvars.iv.next56, 8
  br i1 %exitcond57.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:                                        ; preds = %for.cond5.preheader, %for.body8
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %c.051 = phi double [ 0.000000e+00, %for.cond5.preheader ], [ %add, %for.body8 ]
  %arrayidx10 = getelementptr inbounds [64 x double], ptr %A, i64 %indvars.iv58, i64 %indvars.iv
  %1 = load double, ptr %arrayidx10, align 8
  %arrayidx14 = getelementptr inbounds [64 x double], ptr %B, i64 %indvars.iv55, i64 %indvars.iv
  %2 = load double, ptr %arrayidx14, align 8
  %mul = fmul double %1, %2
  %add = fadd double %c.051, %mul
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8
}
