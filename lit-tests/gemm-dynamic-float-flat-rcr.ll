; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --scrub-attributes
; RUN: opt -passes='kernel-replacer-pass,loop-deletion,dce,simplifycfg,loop-deletion,dce,simplifycfg,dce' --kernelfarer-replacement-mode=cblas-interface -S < %s | FileCheck %s

define void @_Z14dynamicSizeGEMMiiiPKfiS0_iPfiff(i32 %m, i32 %n, i32 %k, ptr %A, i32 %lda, ptr %B, i32 %ldb, ptr %C, i32 %ldc, float %alpha, float %beta) {
; CHECK-LABEL: @_Z14dynamicSizeGEMMiiiPKfiS0_iPfiff(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @cblas_sgemm(i32 101, i32 111, i32 112, i32 [[M:%.*]], i32 [[N:%.*]], i32 [[K:%.*]], float [[ALPHA:%.*]], ptr [[A:%.*]], i32 [[LDA:%.*]], ptr [[B:%.*]], i32 [[LDB:%.*]], float [[BETA:%.*]], ptr [[C:%.*]], i32 [[LDC:%.*]])
; CHECK-NEXT:    ret void
;
entry:
  %cmp58 = icmp sgt i32 %m, 0
  br i1 %cmp58, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp255 = icmp sgt i32 %n, 0
  %cmp652 = icmp sgt i32 %k, 0
  %0 = sext i32 %ldb to i64
  %1 = sext i32 %lda to i64
  %2 = sext i32 %ldc to i64
  %wide.trip.count73 = zext i32 %m to i64
  %wide.trip.count67 = zext i32 %n to i64
  %wide.trip.count = zext i32 %k to i64
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.lr.ph, %for.cond.cleanup3
  %indvars.iv69 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next70, %for.cond.cleanup3 ]
  br i1 %cmp255, label %for.cond5.preheader.lr.ph, label %for.cond.cleanup3

for.cond5.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %3 = mul nsw i64 %indvars.iv69, %1
  %4 = mul nsw i64 %indvars.iv69, %2
  br label %for.cond5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void

for.cond5.preheader:                              ; preds = %for.cond5.preheader.lr.ph, %for.cond.cleanup7
  %indvars.iv63 = phi i64 [ 0, %for.cond5.preheader.lr.ph ], [ %indvars.iv.next64, %for.cond.cleanup7 ]
  br i1 %cmp652, label %for.body8.lr.ph, label %for.cond.cleanup7

for.body8.lr.ph:                                  ; preds = %for.cond5.preheader
  %5 = mul nsw i64 %indvars.iv63, %0
  br label %for.body8

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %indvars.iv.next70 = add nuw nsw i64 %indvars.iv69, 1
  %exitcond74.not = icmp eq i64 %indvars.iv.next70, %wide.trip.count73
  br i1 %exitcond74.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup7:                                ; preds = %for.body8, %for.cond5.preheader
  %c.0.lcssa = phi float [ 0.000000e+00, %for.cond5.preheader ], [ %add14, %for.body8 ]
  %mul15 = fmul float %c.0.lcssa, %alpha
  %6 = add nsw i64 %indvars.iv63, %4
  %arrayidx19 = getelementptr inbounds float, ptr %C, i64 %6
  %7 = load float, ptr %arrayidx19, align 4
  %mul20 = fmul float %7, %beta
  %add21 = fadd float %mul15, %mul20
  store float %add21, ptr %arrayidx19, align 4
  %indvars.iv.next64 = add nuw nsw i64 %indvars.iv63, 1
  %exitcond68.not = icmp eq i64 %indvars.iv.next64, %wide.trip.count67
  br i1 %exitcond68.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:                                        ; preds = %for.body8.lr.ph, %for.body8
  %indvars.iv = phi i64 [ 0, %for.body8.lr.ph ], [ %indvars.iv.next, %for.body8 ]
  %c.053 = phi float [ 0.000000e+00, %for.body8.lr.ph ], [ %add14, %for.body8 ]
  %8 = add nsw i64 %indvars.iv, %3
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %8
  %9 = load float, ptr %arrayidx, align 4
  %10 = add nsw i64 %indvars.iv, %5
  %arrayidx12 = getelementptr inbounds float, ptr %B, i64 %10
  %11 = load float, ptr %arrayidx12, align 4
  %mul13 = fmul float %9, %11
  %add14 = fadd float %c.053, %mul13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8
}
