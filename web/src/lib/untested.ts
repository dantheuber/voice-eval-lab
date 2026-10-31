import type { CallArtifact, Profile } from "@/api/types";

/* Whether a call or run went through adapters that were never tested against a live provider.
   Prefer the flag the backend stores with the call or run; fall back to the artifact's recorded providers,
   then to the profile list. Bedrock adapters are the untested ones on this machine. */
export function isUntested(
  x: { untested?: boolean; profile: string | null; providers?: CallArtifact["providers"] } | null | undefined,
  profiles?: Profile[],
): boolean {
  if (!x) return false;
  if (typeof x.untested === "boolean") return x.untested;
  if (x.providers && Object.values(x.providers).some((p) => /^bedrock/i.test(p))) return true;
  const prof = profiles?.find((p) => p.id === x.profile);
  return !!prof?.untested;
}
