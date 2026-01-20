-- SECURITY PATCH V1: ARMOR PLATING
-- Act Enable RLS and secure policies for all exposed tables
-- 1. Enable RLS on sensitive tables
ALTER TABLE "public"."bot_activity_log" ENABLE ROW LEVEL SECURITY;

ALTER TABLE "public"."live_metrics" ENABLE ROW LEVEL SECURITY;

ALTER TABLE "public"."neural_evolution" ENABLE ROW LEVEL SECURITY;

ALTER TABLE "public"."neural_signals" ENABLE ROW LEVEL SECURITY;

ALTER TABLE "public"."neural_training_history" ENABLE ROW LEVEL SECURITY;

-- 2. Drop existing permissive policies (if any) to start fresh
DROP POLICY IF EXISTS "Enable read access for all users" ON "public"."bot_activity_log";

DROP POLICY IF EXISTS "allow_anon_insert_live_metrics" ON "public"."live_metrics";

DROP POLICY IF EXISTS "allow_anon_update_live_metrics" ON "public"."live_metrics";

DROP POLICY IF EXISTS "Allow anonymous insert" ON "public"."neural_evolution";

DROP POLICY IF EXISTS "allow_anon_insert_neural_signals" ON "public"."neural_signals";

DROP POLICY IF EXISTS "Allow bot access" ON "public"."neural_training_history";

-- 3. Create Restrictive Policies
-- bot_activity_log:
-- Allow Service Role (Backend) to INSERT/UPDATE/DELETE.
-- Allow Anon (Dashboard) to SELECT (View logs).
CREATE POLICY "Service Role Full Access" ON "public"."bot_activity_log" FOR ALL USING (auth.role () = 'service_role')
WITH
    CHECK (auth.role () = 'service_role');

CREATE POLICY "Public Read Access" ON "public"."bot_activity_log" FOR
SELECT
    USING (true);

-- live_metrics:
-- Strictly Backend write, Public read (for dashboard charts).
CREATE POLICY "Service Role Write Metrics" ON "public"."live_metrics" FOR INSERT
WITH
    CHECK (auth.role () = 'service_role');

CREATE POLICY "Service Role Update Metrics" ON "public"."live_metrics" FOR
UPDATE USING (auth.role () = 'service_role')
WITH
    CHECK (auth.role () = 'service_role');

CREATE POLICY "Public Read Metrics" ON "public"."live_metrics" FOR
SELECT
    USING (true);

-- neural_evolution:
-- Strictly Backend write, Public read.
CREATE POLICY "Service Role Write Evolution" ON "public"."neural_evolution" FOR INSERT
WITH
    CHECK (auth.role () = 'service_role');

CREATE POLICY "Public Read Evolution" ON "public"."neural_evolution" FOR
SELECT
    USING (true);

-- neural_signals:
-- Strictly Backend write, Public read.
CREATE POLICY "Service Role Write Signals" ON "public"."neural_signals" FOR INSERT
WITH
    CHECK (auth.role () = 'service_role');

CREATE POLICY "Public Read Signals" ON "public"."neural_signals" FOR
SELECT
    USING (true);

-- neural_training_history:
-- Strictly Backend write, Public read.
CREATE POLICY "Service Role Write History" ON "public"."neural_training_history" FOR INSERT
WITH
    CHECK (auth.role () = 'service_role');

CREATE POLICY "Public Read History" ON "public"."neural_training_history" FOR
SELECT
    USING (true);

-- 4. Fix Function Search Paths (Security Best Practice)
ALTER FUNCTION public.update_learning_confidence (jsonb)
SET
    search_path = public;

-- ALTER FUNCTION public.get_asset_profile(text) SET search_path = public; -- Assuming this function exists based on logs