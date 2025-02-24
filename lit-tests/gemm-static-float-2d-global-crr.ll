; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --scrub-attributes
; RUN: opt -passes='gemm-replacer-pass,loop-deletion,dce,simplifycfg,loop-deletion,dce,simplifycfg' --gemmfarer-replacement-mode=matrix-intrinsics -S < %s | FileCheck %s

@A = global [64 x [16 x float]] zeroinitializer, align 16
@B = global [64 x [8 x float]] zeroinitializer, align 16
@C = global [16 x [8 x float]] zeroinitializer, align 16

define void @_Z14staticSizeGEMMff(float %alpha, float %beta) {
; CHECK-LABEL: @_Z14staticSizeGEMMff(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call <1024 x float> @llvm.matrix.column.major.load.v1024f32.i64(ptr align 4 @A, i64 16, i1 false, i32 16, i32 64)
; CHECK-NEXT:    [[TMP1:%.*]] = call <512 x float> @llvm.matrix.column.major.load.v512f32.i64(ptr align 4 @B, i64 8, i1 false, i32 8, i32 64)
; CHECK-NEXT:    [[TMP2:%.*]] = call <512 x float> @llvm.matrix.transpose.v512f32(<512 x float> [[TMP1]], i32 8, i32 64)
; CHECK-NEXT:    [[TMP3:%.*]] = call <128 x float> @llvm.matrix.multiply.v128f32.v1024f32.v512f32(<1024 x float> [[TMP0]], <512 x float> [[TMP2]], i32 16, i32 64, i32 8)
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLATINSERT:%.*]] = insertelement <128 x float> poison, float [[ALPHA:%.*]], i32 0
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLAT:%.*]] = shufflevector <128 x float> [[SCALAR_SPLAT_SPLATINSERT]], <128 x float> poison, <128 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = fmul <128 x float> [[SCALAR_SPLAT_SPLAT]], [[TMP3]]
; CHECK-NEXT:    [[TMP5:%.*]] = call <128 x float> @llvm.matrix.column.major.load.v128f32.i64(ptr align 4 @C, i64 8, i1 false, i32 8, i32 16)
; CHECK-NEXT:    [[TMP6:%.*]] = call <128 x float> @llvm.matrix.transpose.v128f32(<128 x float> [[TMP5]], i32 8, i32 16)
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLATINSERT1:%.*]] = insertelement <128 x float> poison, float [[BETA:%.*]], i32 0
; CHECK-NEXT:    [[SCALAR_SPLAT_SPLAT2:%.*]] = shufflevector <128 x float> [[SCALAR_SPLAT_SPLATINSERT1]], <128 x float> poison, <128 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = fmul <128 x float> [[SCALAR_SPLAT_SPLAT2]], [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = fadd <128 x float> [[TMP4]], [[TMP7]]
; CHECK-NEXT:    [[TMP9:%.*]] = call <128 x float> @llvm.matrix.transpose.v128f32(<128 x float> [[TMP8]], i32 16, i32 8)
; CHECK-NEXT:    call void @llvm.matrix.column.major.store.v128f32.i64(<128 x float> [[TMP9]], ptr align 4 @C, i64 8, i1 false, i32 8, i32 16)
; CHECK-NEXT:    ret void
;
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %indvars.iv57 = phi i64 [ 0, %entry ], [ %indvars.iv.next58, %for.cond.cleanup3 ]
  br label %for.cond5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void

for.cond5.preheader:                              ; preds = %for.cond1.preheader, %for.cond.cleanup7
  %indvars.iv54 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next55, %for.cond.cleanup7 ]
  br label %for.body8

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %indvars.iv.next58 = add nuw nsw i64 %indvars.iv57, 1
  %exitcond59.not = icmp eq i64 %indvars.iv.next58, 16
  br i1 %exitcond59.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup7:                                ; preds = %for.body8
  %mul15 = fmul float %add, %alpha
  %arrayidx19 = getelementptr inbounds [16 x [8 x float]], ptr @C, i64 0, i64 %indvars.iv57, i64 %indvars.iv54
  %0 = load float, ptr %arrayidx19, align 4
  %mul20 = fmul float %0, %beta
  %add21 = fadd float %mul15, %mul20
  store float %add21, ptr %arrayidx19, align 4
  %indvars.iv.next55 = add nuw nsw i64 %indvars.iv54, 1
  %exitcond56.not = icmp eq i64 %indvars.iv.next55, 8
  br i1 %exitcond56.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:                                        ; preds = %for.cond5.preheader, %for.body8
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %c.050 = phi float [ 0.000000e+00, %for.cond5.preheader ], [ %add, %for.body8 ]
  %arrayidx10 = getelementptr inbounds [64 x [16 x float]], ptr @A, i64 0, i64 %indvars.iv, i64 %indvars.iv57
  %1 = load float, ptr %arrayidx10, align 4
  %arrayidx14 = getelementptr inbounds [64 x [8 x float]], ptr @B, i64 0, i64 %indvars.iv, i64 %indvars.iv54
  %2 = load float, ptr %arrayidx14, align 4
  %mul = fmul float %1, %2
  %add = fadd float %c.050, %mul
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8
}
