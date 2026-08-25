-- ─────────────────────────────────────────────────────────────
-- SCRIPT DE MIGRACIÓN Y CREACIÓN DE TABLAS EN SUPABASE
-- Ejecutar este script en el SQL Editor de tu Dashboard de Supabase
-- ─────────────────────────────────────────────────────────────

-- 1. Crear extensión UUID si no existe
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Crear tabla chat_logs para persistir cada interacción del bot
CREATE TABLE IF NOT EXISTS public.chat_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_query TEXT NOT NULL,
    assistant_response TEXT,
    unidad INT,
    sources JSONB DEFAULT '[]'::jsonb,
    response_time_ms INT,
    status TEXT NOT NULL DEFAULT 'success', -- 'success', 'error', 'blocked_guardrails'
    error_message TEXT,
    model_used TEXT,
    client_ip TEXT
);

-- 3. Crear índices optimizados para estadísticas y filtros rápidos
CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at ON public.chat_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_logs_unidad ON public.chat_logs (unidad);
CREATE INDEX IF NOT EXISTS idx_chat_logs_status ON public.chat_logs (status);

-- 4. Habilitar RLS (Row Level Security) y permitir inserciones de la API anon/service_role
ALTER TABLE public.chat_logs ENABLE ROW LEVEL SECURITY;

-- Política para permitir inserciones públicas/API
CREATE POLICY "Permitir inserciones desde API" 
ON public.chat_logs 
FOR INSERT 
WITH CHECK (true);

-- Política para permitir lectura desde la API
CREATE POLICY "Permitir lectura desde API" 
ON public.chat_logs 
FOR SELECT 
USING (true);
