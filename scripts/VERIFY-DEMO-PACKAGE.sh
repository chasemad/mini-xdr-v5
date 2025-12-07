#!/bin/bash
# Demo Package Verification Script

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Mini-XDR Demo Package Verification                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

check_file() {
    if [ -f "$1" ]; then
        echo "✅ $1"
        return 0
    else
        echo "❌ $1 (MISSING)"
        return 1
    fi
}

check_executable() {
    if [ -x "$1" ]; then
        echo "✅ $1 (executable)"
        return 0
    else
        echo "⚠️  $1 (not executable)"
        return 1
    fi
}

MISSING=0

echo "📄 Main Documentation:"
check_file "DEMO-PACKAGE-INDEX.md" || MISSING=$((MISSING+1))
check_file "DEMO-READY.md" || MISSING=$((MISSING+1))
echo ""

echo "📁 Demo Scripts:"
check_executable "scripts/demo/pre-demo-setup.sh" || MISSING=$((MISSING+1))
check_executable "scripts/demo/demo-attack.sh" || MISSING=$((MISSING+1))
check_executable "scripts/demo/manual-event-injection.sh" || MISSING=$((MISSING+1))
check_executable "scripts/demo/validate-demo-ready.sh" || MISSING=$((MISSING+1))
echo ""

echo "📚 Demo Documentation:"
check_file "scripts/demo/README.md" || MISSING=$((MISSING+1))
check_file "scripts/demo/DEMO-WALKTHROUGH.md" || MISSING=$((MISSING+1))
check_file "scripts/demo/demo-cheatsheet.md" || MISSING=$((MISSING+1))
check_file "scripts/demo/QUICK-REFERENCE.txt" || MISSING=$((MISSING+1))
echo ""

if [ $MISSING -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              ✅ ALL FILES PRESENT AND READY!                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🎬 You're ready to create your demo!"
    echo ""
    echo "Next steps:"
    echo "  1. Read: DEMO-PACKAGE-INDEX.md"
    echo "  2. Run:  ./scripts/demo/pre-demo-setup.sh"
    echo "  3. Run:  ./scripts/demo/validate-demo-ready.sh"
    echo "  4. Record your demo!"
    exit 0
else
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           ⚠️  $MISSING FILES MISSING OR NOT EXECUTABLE       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    exit 1
fi
