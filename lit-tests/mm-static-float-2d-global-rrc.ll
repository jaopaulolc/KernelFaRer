; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --scrub-attributes
; RUN: opt -passes='kernel-replacer-pass,loop-deletion,dce,simplifycfg' --kernelfarer-replacement-mode=matrix-intrinsics -S < %s | FileCheck %s

@A = global [16 x [64 x float]] zeroinitializer, align 16
@B = global [64 x [8 x float]] zeroinitializer, align 16
@C = global [8 x [16 x float]] zeroinitializer, align 16

define void @_Z11fixedSizeMMv() {
; CHECK-LABEL: @_Z11fixedSizeMMv(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call <1024 x float> @llvm.matrix.column.major.load.v1024f32.i64(ptr align 4 @A, i64 64, i1 false, i32 64, i32 16)
; CHECK-NEXT:    [[TMP1:%.*]] = call <1024 x float> @llvm.matrix.transpose.v1024f32(<1024 x float> [[TMP0]], i32 64, i32 16)
; CHECK-NEXT:    [[TMP2:%.*]] = call <512 x float> @llvm.matrix.column.major.load.v512f32.i64(ptr align 4 @B, i64 8, i1 false, i32 8, i32 64)
; CHECK-NEXT:    [[TMP3:%.*]] = call <512 x float> @llvm.matrix.transpose.v512f32(<512 x float> [[TMP2]], i32 8, i32 64)
; CHECK-NEXT:    [[TMP4:%.*]] = call <128 x float> @llvm.matrix.multiply.v128f32.v1024f32.v512f32(<1024 x float> [[TMP1]], <512 x float> [[TMP3]], i32 16, i32 64, i32 8)
; CHECK-NEXT:    call void @llvm.matrix.column.major.store.v128f32.i64(<128 x float> [[TMP4]], ptr align 4 @C, i64 16, i1 false, i32 16, i32 8)
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
  %arrayidx18 = getelementptr inbounds [8 x [16 x float]], ptr @C, i64 0, i64 %indvars.iv45, i64 %indvars.iv48
  store float %add, ptr %arrayidx18, align 4
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond47.not = icmp eq i64 %indvars.iv.next46, 8
  br i1 %exitcond47.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:                                        ; preds = %for.cond5.preheader, %for.body8
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %c.041 = phi float [ 0.000000e+00, %for.cond5.preheader ], [ %add, %for.body8 ]
  %arrayidx10 = getelementptr inbounds [16 x [64 x float]], ptr @A, i64 0, i64 %indvars.iv48, i64 %indvars.iv
  %0 = load float, ptr %arrayidx10, align 4
  %arrayidx14 = getelementptr inbounds [64 x [8 x float]], ptr @B, i64 0, i64 %indvars.iv, i64 %indvars.iv45
  %1 = load float, ptr %arrayidx14, align 4
  %mul = fmul float %0, %1
  %add = fadd float %c.041, %mul
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8
}
