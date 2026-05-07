// SwiftPM module marker for CNumKong.
//
// CNumKong is the header-only module facade for NumKong's public C API; the
// actual runtime dispatch implementation lives in the CNumKongDispatch target
// (see ../c/numkong.c and ../c/dispatch_*.c).
//
// SwiftPM/Xcode 16+ refuses to link a C target that produces no object files
// when it appears as a transitive dependency of an app target — the link step
// expects a CNumKong.o output that would otherwise never exist
// (see swiftlang/swift-package-manager#5706, open since October 2022).
//
// This file provides one real exported symbol so the relocatable link
// succeeds. The same pattern is used by apple/swift-system's
// Sources/CSystem/shims.c. There is no runtime cost — the symbol is read once
// at most by tooling that needs to verify the module is present.
#include "numkong/numkong.h"

const unsigned int nk_cnumkong_module_loaded = 1u;
