#!/bin/bash
# install_model.sh - Instala Ollama y baja el mejor modelo que cabe en la RAM disponible
# Uso: bash scripts/install_model.sh [modelo_específico]

set -e

echo ""
echo "======================================================"
echo "  CatanLLM - Instalación del modelo LLM"
echo "======================================================"

# ── 1. Instalar Ollama si no está ──────────────────────────────────────────
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "📦 Instalando Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama instalado."
else
    echo "✅ Ollama ya está instalado: $(ollama --version)"
fi

# ── 2. Arrancar Ollama ─────────────────────────────────────────────────────
echo ""
echo "🚀 Arrancando servidor Ollama..."

# Matar instancia previa si existe
pkill ollama 2>/dev/null || true
sleep 1

# Arrancar en background
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "   PID: $OLLAMA_PID"
sleep 4

# Verificar que arrancó
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama servidor activo en localhost:11434"
else
    echo "❌ No se pudo conectar con Ollama. Revisa /tmp/ollama.log"
    cat /tmp/ollama.log
    exit 1
fi

# ── 3. Seleccionar modelo según RAM disponible ─────────────────────────────
echo ""
echo "🔍 Analizando RAM disponible..."
AVAILABLE_MB=$(free -m | awk '/^Mem:/ {print $7}')
TOTAL_MB=$(free -m | awk '/^Mem:/ {print $2}')
echo "   RAM disponible: ${AVAILABLE_MB}MB / ${TOTAL_MB}MB total"

# Modelo pasado como argumento
if [ -n "$1" ]; then
    MODEL="$1"
    echo "   Usando modelo especificado: $MODEL"
elif [ "$AVAILABLE_MB" -gt 6000 ]; then
    MODEL="qwen2.5:7b-instruct-q4_K_M"
    echo "   RAM suficiente para 7B Q4 (~5.2GB)"
elif [ "$AVAILABLE_MB" -gt 4000 ]; then
    MODEL="qwen2.5:7b-instruct-q2_K"
    echo "   RAM moderada, usando 7B Q2 (~2.9GB)"
elif [ "$AVAILABLE_MB" -gt 2500 ]; then
    MODEL="qwen2.5:3b-instruct"
    echo "   RAM limitada, usando 3B (~1.9GB) — buena calidad para el tamaño"
else
    MODEL="qwen2.5:1.5b-instruct"
    echo "   RAM muy limitada, usando 1.5B (~1GB)"
fi

# ── 4. Bajar el modelo ─────────────────────────────────────────────────────
echo ""
echo "📥 Bajando modelo: $MODEL"
echo "   (Esto puede tardar varios minutos la primera vez...)"

if ollama pull "$MODEL"; then
    echo "✅ Modelo $MODEL listo."
else
    echo "⚠️  No se pudo bajar $MODEL, intentando con qwen2.5:3b..."
    MODEL="qwen2.5:3b"
    ollama pull "$MODEL"
    echo "✅ Modelo fallback $MODEL listo."
fi

# ── 5. Test rápido ─────────────────────────────────────────────────────────
echo ""
echo "🧪 Test rápido del modelo..."
echo ""

TEST_RESPONSE=$(echo 'You are playing Catan. You have 2 wood and 1 clay. Should you build a road? Reply ONLY with JSON: {"action": "road", "node_id": 5, "road_to": 6}' | ollama run "$MODEL" 2>/dev/null)

echo "   Prompt: 'Should you build a road?'"
echo "   Response: $TEST_RESPONSE"

if echo "$TEST_RESPONSE" | grep -q '"action"'; then
    echo "✅ El modelo responde con JSON válido."
else
    echo "⚠️  El modelo respondió pero no con JSON puro. Verifica los prompts."
fi

# ── 6. Actualizar configuración ────────────────────────────────────────────
echo ""
echo "📝 Actualizando configuración en CatanLLM..."

CONFIG_FILE="$(dirname "$0")/../config.json"
cat > "$CONFIG_FILE" <<EOF
{
  "model": "$MODEL",
  "ollama_url": "http://localhost:11434",
  "timeout": 30,
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "   Guardado en config.json: modelo = $MODEL"

# ── 7. Resumen ─────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  ✅ INSTALACIÓN COMPLETA"
echo "======================================================"
echo "  Modelo:  $MODEL"
echo "  RAM usada: ~$(du -sh ~/.ollama/models 2>/dev/null | cut -f1 || echo '?')"
echo ""
echo "  Para jugar una partida:"
echo "    python scripts/run_game.py --model $MODEL"
echo ""
echo "  Para dry-run (sin LLM, solo prueba la integración):"
echo "    python scripts/run_game.py --dry-run"
echo "======================================================"
