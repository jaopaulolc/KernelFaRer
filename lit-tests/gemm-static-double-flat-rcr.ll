; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --scrub-attributes
; RUN: opt -passes='kernel-replacer-pass,loop-deletion,dce,simplifycfg,loop-deletion,dce,simplifycfg' --kernelfarer-replacement-mode=matrix-intrinsics -S < %s | FileCheck %s

define void @_Z14staticSizeGEMMPKdiS0_iPdidd(ptr %A, i32 %lda, ptr %B, i32 %ldb, ptr %C, i32 %ldc, double %alpha, double %beta) {
; CHECK-LABEL: @_Z14staticSizeGEMMPKdiS0_iPdidd(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = sext i32 [[LDA:%.*]] to i64
; CHECK-NEXT:    [[TMP1:%.*]] = sext i32 [[LDB:%.*]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = sext i32 [[LDC:%.*]] to i64
; CHECK-NEXT:    [[TMP3:%.*]] = call <1024 x double> @llvm.matrix.column.major.load.v1024f64.i64(ptr align 8 [[A:%.*]], i64 [[TMP0]], i1 false, i32 64, i32 16)
; CHECK-NEXT:    [[TMP4:%.*]] = call <1024 x double> @llvm.matrix.transpose.v1024f64(<1024 x double> [[TMP3]], i32 64, i32 16)
; CHECK-NEXT:    [[TMP5:%.*]] = call <512 x double> @llvm.matrix.column.major.load.v512f64.i64(ptr align 8 [[B:%.*]], i64 [[TMP1]], i1 false, i32 64, i32 8)
; CHECK-NEXT:    [[TMP6:%.*]] = call <128 x double> @llvm.matrix.multiply.v128f64.v1024f64.v512f64(<1024 x double> [[TMP4]], <512 x double> [[TMP5]], i32 16, i32 64, i32 8)
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLATINSERT:%.*]] = insertelement <128 x double> poison, double [[ALPHA:%.*]], i64 0
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLAT:%.*]] = shufflevector <128 x double> [[SCALAR_SPLAT_SPLATINSERT]], <128 x double> poison, <128 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = fmul <128 x double> [[SCALAR_SPLAT_SPLAT]], [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = call <128 x double> @llvm.matrix.column.major.load.v128f64.i64(ptr align 8 [[C:%.*]], i64 [[TMP2]], i1 false, i32 8, i32 16)
; CHECK-NEXT:    [[TMP9:%.*]] = call <128 x double> @llvm.matrix.transpose.v128f64(<128 x double> [[TMP8]], i32 8, i32 16)
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLATINSERT1:%.*]] = insertelement <128 x double> poison, double [[BETA:%.*]], i64 0
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLAT2:%.*]] = shufflevector <128 x double> [[SCALAR_SPLAT_SPLATINSERT1]], <128 x double> poison, <128 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP10:%.*]] = fmul <128 x double> [[SCALAR_SPLAT_SPLAT2]], [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = fadd <128 x double> [[TMP7]], [[TMP10]]
; CHECK-NEXT:    [[TMP12:%.*]] = call <128 x double> @llvm.matrix.transpose.v128f64(<128 x double> [[TMP11]], i32 16, i32 8)
; CHECK-NEXT:    call void @llvm.matrix.column.major.store.v128f64.i64(<128 x double> [[TMP12]], ptr align 8 [[C]], i64 [[TMP2]], i1 false, i32 8, i32 16)
; CHECK-NEXT:    ret void
;
entry:
  %0 = sext i32 %ldb to i64
  %1 = sext i32 %lda to i64
  %2 = sext i32 %ldc to i64
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %indvars.iv66 = phi i64 [ 0, %entry ], [ %indvars.iv.next67, %for.cond.cleanup3 ]
  %3 = mul nsw i64 %indvars.iv66, %1
  %4 = mul nsw i64 %indvars.iv66, %2
  br label %for.cond5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void

for.cond5.preheader:                              ; preds = %for.cond1.preheader, %for.cond.cleanup7
  %indvars.iv61 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next62, %for.cond.cleanup7 ]
  %5 = mul nsw i64 %indvars.iv61, %0
  br label %for.body8

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %indvars.iv.next67 = add nuw nsw i64 %indvars.iv66, 1
  %exitcond70.not = icmp eq i64 %indvars.iv.next67, 16
  br i1 %exitcond70.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup7:                                ; preds = %for.body8
  %mul15 = fmul double %add14, %alpha
  %6 = add nsw i64 %indvars.iv61, %4
  %arrayidx19 = getelementptr inbounds double, ptr %C, i64 %6
  %7 = load double, ptr %arrayidx19, align 8
  %mul20 = fmul double %7, %beta
  %add21 = fadd double %mul15, %mul20
  store double %add21, ptr %arrayidx19, align 8
  %indvars.iv.next62 = add nuw nsw i64 %indvars.iv61, 1
  %exitcond65.not = icmp eq i64 %indvars.iv.next62, 8
  br i1 %exitcond65.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:                                        ; preds = %for.cond5.preheader, %for.body8
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %c.055 = phi double [ 0.000000e+00, %for.cond5.preheader ], [ %add14, %for.body8 ]
  %8 = add nsw i64 %indvars.iv, %3
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %8
  %9 = load double, ptr %arrayidx, align 8
  %10 = add nsw i64 %indvars.iv, %5
  %arrayidx12 = getelementptr inbounds double, ptr %B, i64 %10
  %11 = load double, ptr %arrayidx12, align 8
  %mul13 = fmul double %9, %11
  %add14 = fadd double %c.055, %mul13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8
}
