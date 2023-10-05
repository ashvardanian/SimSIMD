	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 3
	.globl	_simsimd_neon_f32_l2sq          ; -- Begin function simsimd_neon_f32_l2sq
	.p2align	2
_simsimd_neon_f32_l2sq:                 ; @simsimd_neon_f32_l2sq
	.cfi_startproc
; %bb.0:
	cmp	x2, #4
	b.hs	LBB0_2
; %bb.1:
	mov	x11, #0
	movi.2d	v0, #0000000000000000
	b	LBB0_4
LBB0_2:
	mov	x9, #0
	movi.2d	v0, #0000000000000000
	mov	x8, x1
	mov	x10, x0
LBB0_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	q1, [x10], #16
	ldr	q2, [x8], #16
	fsub.4s	v1, v1, v2
	fmla.4s	v0, v1, v1
	add	x11, x9, #4
	add	x12, x9, #8
	mov	x9, x11
	cmp	x12, x2
	b.ls	LBB0_3
LBB0_4:
	faddp.4s	v0, v0, v0
	faddp.2s	s0, v0
	cmp	x11, x2
	b.hs	LBB0_12
; %bb.5:
	sub	x9, x2, x11
	cmp	x9, #8
	b.hs	LBB0_7
; %bb.6:
	mov	x8, x11
	b	LBB0_10
LBB0_7:
	and	x10, x9, #0xfffffffffffffff8
	add	x8, x11, x10
	movi.2d	v1, #0000000000000000
	mov.s	v1[0], v0[0]
	movi.2d	v0, #0000000000000000
	lsl	x11, x11, #2
	add	x12, x11, #16
	add	x11, x0, x12
	add	x12, x1, x12
	mov	x13, x10
LBB0_8:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q2, q3, [x11, #-16]
	ldp	q4, q5, [x12, #-16]
	fsub.4s	v2, v2, v4
	fsub.4s	v3, v3, v5
	fmla.4s	v1, v2, v2
	fmla.4s	v0, v3, v3
	add	x11, x11, #32
	add	x12, x12, #32
	subs	x13, x13, #8
	b.ne	LBB0_8
; %bb.9:
	fadd.4s	v0, v0, v1
	faddp.4s	v0, v0, v0
	faddp.2s	s0, v0
	cmp	x9, x10
	b.eq	LBB0_12
LBB0_10:
	sub	x9, x2, x8
	lsl	x10, x8, #2
	add	x8, x1, x10
	add	x10, x0, x10
LBB0_11:                                ; =>This Inner Loop Header: Depth=1
	ldr	s1, [x10], #4
	ldr	s2, [x8], #4
	fsub	s1, s1, s2
	fmadd	s0, s1, s1, s0
	subs	x9, x9, #1
	b.ne	LBB0_11
LBB0_12:
                                        ; kill: def $s0 killed $s0 killed $q0
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_simsimd_neon_f32_ip            ; -- Begin function simsimd_neon_f32_ip
	.p2align	2
_simsimd_neon_f32_ip:                   ; @simsimd_neon_f32_ip
	.cfi_startproc
; %bb.0:
	cmp	x2, #4
	b.hs	LBB1_2
; %bb.1:
	mov	x11, #0
	movi.2d	v0, #0000000000000000
	b	LBB1_4
LBB1_2:
	mov	x9, #0
	movi.2d	v0, #0000000000000000
	mov	x8, x1
	mov	x10, x0
LBB1_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	q1, [x10], #16
	ldr	q2, [x8], #16
	fmla.4s	v0, v2, v1
	add	x11, x9, #4
	add	x12, x9, #8
	mov	x9, x11
	cmp	x12, x2
	b.ls	LBB1_3
LBB1_4:
	faddp.4s	v0, v0, v0
	faddp.2s	s0, v0
	cmp	x11, x2
	b.hs	LBB1_12
; %bb.5:
	sub	x9, x2, x11
	cmp	x9, #8
	b.hs	LBB1_7
; %bb.6:
	mov	x8, x11
	b	LBB1_10
LBB1_7:
	and	x10, x9, #0xfffffffffffffff8
	add	x8, x11, x10
	movi.2d	v1, #0000000000000000
	mov.s	v1[0], v0[0]
	movi.2d	v0, #0000000000000000
	lsl	x11, x11, #2
	add	x12, x11, #16
	add	x11, x0, x12
	add	x12, x1, x12
	mov	x13, x10
LBB1_8:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q2, q3, [x11, #-16]
	ldp	q4, q5, [x12, #-16]
	fmla.4s	v1, v4, v2
	fmla.4s	v0, v5, v3
	add	x11, x11, #32
	add	x12, x12, #32
	subs	x13, x13, #8
	b.ne	LBB1_8
; %bb.9:
	fadd.4s	v0, v0, v1
	faddp.4s	v0, v0, v0
	faddp.2s	s0, v0
	cmp	x9, x10
	b.eq	LBB1_12
LBB1_10:
	sub	x9, x2, x8
	lsl	x10, x8, #2
	add	x8, x1, x10
	add	x10, x0, x10
LBB1_11:                                ; =>This Inner Loop Header: Depth=1
	ldr	s1, [x10], #4
	ldr	s2, [x8], #4
	fmadd	s0, s2, s1, s0
	subs	x9, x9, #1
	b.ne	LBB1_11
LBB1_12:
	fmov	s1, #1.00000000
	fsub	s0, s1, s0
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_simsimd_neon_f32_cos           ; -- Begin function simsimd_neon_f32_cos
	.p2align	2
_simsimd_neon_f32_cos:                  ; @simsimd_neon_f32_cos
	.cfi_startproc
; %bb.0:
	cmp	x2, #4
	b.hs	LBB2_2
; %bb.1:
	mov	x11, #0
	movi.2d	v0, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v1, #0000000000000000
	b	LBB2_4
LBB2_2:
	mov	x9, #0
	movi.2d	v1, #0000000000000000
	mov	x8, x1
	mov	x10, x0
	movi.2d	v2, #0000000000000000
	movi.2d	v0, #0000000000000000
LBB2_3:                                 ; =>This Inner Loop Header: Depth=1
	ldr	q3, [x10], #16
	ldr	q4, [x8], #16
	fmla.4s	v1, v4, v3
	fmla.4s	v2, v3, v3
	fmla.4s	v0, v4, v4
	add	x11, x9, #4
	add	x12, x9, #8
	mov	x9, x11
	cmp	x12, x2
	b.ls	LBB2_3
LBB2_4:
	faddp.4s	v1, v1, v1
	faddp.2s	s1, v1
	faddp.4s	v2, v2, v2
	faddp.2s	s2, v2
	faddp.4s	v0, v0, v0
	faddp.2s	s0, v0
	cmp	x11, x2
	b.hs	LBB2_12
; %bb.5:
	sub	x9, x2, x11
	cmp	x9, #8
	b.hs	LBB2_7
; %bb.6:
	mov	x8, x11
	b	LBB2_10
LBB2_7:
	and	x10, x9, #0xfffffffffffffff8
	add	x8, x11, x10
	movi.2d	v3, #0000000000000000
	movi.2d	v4, #0000000000000000
	mov.s	v4[0], v1[0]
	movi.2d	v1, #0000000000000000
	mov.s	v1[0], v2[0]
	movi.2d	v2, #0000000000000000
	mov.s	v2[0], v0[0]
	lsl	x11, x11, #2
	add	x12, x11, #16
	add	x11, x0, x12
	add	x12, x1, x12
	mov	x13, x10
	movi.2d	v5, #0000000000000000
	movi.2d	v0, #0000000000000000
LBB2_8:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q6, q7, [x11, #-16]
	ldp	q16, q17, [x12, #-16]
	fmla.4s	v4, v16, v6
	fmla.4s	v3, v17, v7
	fmla.4s	v1, v6, v6
	fmla.4s	v5, v7, v7
	fmla.4s	v2, v16, v16
	fmla.4s	v0, v17, v17
	add	x11, x11, #32
	add	x12, x12, #32
	subs	x13, x13, #8
	b.ne	LBB2_8
; %bb.9:
	fadd.4s	v0, v0, v2
	faddp.4s	v0, v0, v0
	faddp.2s	s0, v0
	fadd.4s	v1, v5, v1
	faddp.4s	v1, v1, v0
	faddp.2s	s2, v1
	fadd.4s	v1, v3, v4
	faddp.4s	v1, v1, v0
	faddp.2s	s1, v1
	cmp	x9, x10
	b.eq	LBB2_12
LBB2_10:
	sub	x9, x2, x8
	lsl	x10, x8, #2
	add	x8, x1, x10
	add	x10, x0, x10
LBB2_11:                                ; =>This Inner Loop Header: Depth=1
	ldr	s3, [x10], #4
	ldr	s4, [x8], #4
	fmadd	s1, s4, s3, s1
	fmadd	s2, s3, s3, s2
	fmadd	s0, s4, s4, s0
	subs	x9, x9, #1
	b.ne	LBB2_11
LBB2_12:
	mov.s	v2[1], v0[0]
	frsqrte.2s	v0, v2
	fmul.2s	v0, v0, v0[1]
	fdiv	s0, s1, s0
	fmov	s1, #1.00000000
	fsub	s0, s1, s0
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
