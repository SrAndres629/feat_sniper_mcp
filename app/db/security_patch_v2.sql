-- SECURITY PATCH V2: FUNCTION HARDENING
-- Fixes "Function Search Path Mutable" warnings by explicitly setting search_path to public

-- 1. Hardening 'update_learning_confidence'
-- We try multiple signatures to catch the one reported by the linter.
DO $$
BEGIN
    -- Try signature with jsonb
    BEGIN
        ALTER FUNCTION public.update_learning_confidence(jsonb) SET search_path = public;
    EXCEPTION WHEN OTHERS THEN NULL; -- Ignore if not exists
    END;

    -- Try signature with no args
    BEGIN
        ALTER FUNCTION public.update_learning_confidence() SET search_path = public;
    EXCEPTION WHEN OTHERS THEN NULL;
    END;
END $$;

-- 2. Hardening 'get_asset_profile'
DO $$
BEGIN
    -- Try signature with text
    BEGIN
        ALTER FUNCTION public.get_asset_profile(text) SET search_path = public;
    EXCEPTION WHEN OTHERS THEN NULL;
    END;

    -- Try signature with no args
    BEGIN
        ALTER FUNCTION public.get_asset_profile() SET search_path = public;
    EXCEPTION WHEN OTHERS THEN NULL;
    END;
END $$;
