/* TanStack Query hooks. Keys are arrays starting with the resource name so invalidation is by prefix. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { endpoints as E } from "./client";
import type * as T from "./types";
import { serialize } from "@/lib/stats";
import { toast } from "@/store/toast";

export const keys = {
  overview: (profile?: string) => ["overview", profile ?? ""] as const,
  calls: (q: T.CallsQuery) => ["calls", q] as const,
  call: (id: string) => ["call", id] as const,
  personas: ["personas"] as const,
  persona: (id: string) => ["persona", id] as const,
  scenarios: ["scenarios"] as const,
  scenario: (id: string) => ["scenario", id] as const,
  customers: ["customers"] as const,
  voices: ["voices"] as const,
  profiles: ["profiles"] as const,
  matrix: (profile?: string) => ["matrix", profile ?? ""] as const,
  runs: ["runs"] as const,
  run: (id: string) => ["run", id] as const,
  roles: ["roles"] as const,
  health: ["health"] as const,
  system: ["system"] as const,
  thresholds: ["thresholds"] as const,
  aws: ["aws"] as const,
  compare: (a: string, b: string) => ["compare", a, b] as const,
};

export const useOverview = (profile?: string) => useQuery({ queryKey: keys.overview(profile), queryFn: () => E.overview(profile), refetchInterval: 15_000 });
export const useCalls = (q: T.CallsQuery) => useQuery({ queryKey: keys.calls(q), queryFn: () => E.calls(q), placeholderData: (prev) => prev });
export const useCall = (id: string | undefined) => useQuery({ queryKey: keys.call(id ?? ""), queryFn: () => E.call(id!), enabled: !!id });
export const usePersonas = () => useQuery({ queryKey: keys.personas, queryFn: E.personas });
export const usePersona = (id: string | undefined) => useQuery({ queryKey: keys.persona(id ?? ""), queryFn: () => E.persona(id!), enabled: !!id });
export const useScenarios = () => useQuery({ queryKey: keys.scenarios, queryFn: E.scenarios });
export const useScenario = (id: string | undefined) => useQuery({ queryKey: keys.scenario(id ?? ""), queryFn: () => E.scenario(id!), enabled: !!id });
export const useCustomers = () => useQuery({ queryKey: keys.customers, queryFn: E.customers, staleTime: 60_000 });
export const useVoices = () => useQuery({ queryKey: keys.voices, queryFn: E.voices, staleTime: 60_000 });
export const useProfiles = () => useQuery({ queryKey: keys.profiles, queryFn: E.profiles, staleTime: 30_000 });
export const useMatrix = (profile?: string) => useQuery({ queryKey: keys.matrix(profile), queryFn: () => E.matrix(profile) });
export const useRuns = () => useQuery({ queryKey: keys.runs, queryFn: E.runs });
export const useRun = (id: string | undefined) => useQuery({ queryKey: keys.run(id ?? ""), queryFn: () => E.run(id!), enabled: !!id });
export const useRoles = () => useQuery({ queryKey: keys.roles, queryFn: E.roles, refetchInterval: 5_000 });
export const useHealth = () => useQuery({ queryKey: keys.health, queryFn: E.health, refetchInterval: 5_000 });
export const useSystem = () => useQuery({ queryKey: keys.system, queryFn: E.system, refetchInterval: 15_000 });
export const useThresholds = () => useQuery({ queryKey: keys.thresholds, queryFn: E.thresholds });
export const useAws = () => useQuery({ queryKey: keys.aws, queryFn: E.aws });
export const useCompare = (a?: string, b?: string) => useQuery({ queryKey: keys.compare(a ?? "", b ?? ""), queryFn: () => E.compare(a!, b!), enabled: !!a && !!b });

export function useInvalidate() {
  const qc = useQueryClient();
  return (...prefixes: readonly (readonly unknown[])[]) => prefixes.forEach((p) => qc.invalidateQueries({ queryKey: p as unknown[] }));
}

export function useDecideFinding(callId: string) {
  const qc = useQueryClient();
  // Writes for one call run in order (serialize), the optimistic patch and its rollback touch only the finding
  // concerned, and the server's response replaces the optimistic finding so local state matches what was stored.
  const patchFinding = (findingId: string, fn: (f: T.Finding) => T.Finding) =>
    qc.setQueryData<T.ReviewPayload>(keys.call(callId), (prev) => (prev?.eval ? { ...prev, eval: { ...prev.eval, findings: prev.eval.findings.map((f) => (f.id === findingId ? fn(f) : f)) } } : prev));
  return useMutation({
    mutationFn: ({ findingId, humanVerdict, humanNote }: { findingId: string; humanVerdict: T.HumanVerdict | null; humanNote: string }) =>
      serialize(`decide:${callId}`, () => E.decideFinding(callId, findingId, { humanVerdict, humanNote })),
    onMutate: async (vars) => {
      const prev = qc.getQueryData<T.ReviewPayload>(keys.call(callId))?.eval?.findings.find((f) => f.id === vars.findingId) ?? null;
      patchFinding(vars.findingId, (f) => ({ ...f, humanVerdict: vars.humanVerdict, humanNote: vars.humanNote, humanAt: vars.humanVerdict ? new Date().toISOString() : null }));
      return { prev };
    },
    onSuccess: (saved, vars) => { patchFinding(vars.findingId, () => saved); },
    onError: (e, vars, ctx) => {
      if (ctx?.prev) patchFinding(vars.findingId, () => ctx.prev!);
      toast(`Could not save the decision (${(e as Error).message}). Reverted.`, 4000);
    },
    onSettled: () => { qc.invalidateQueries({ queryKey: ["calls"] }); qc.invalidateQueries({ queryKey: ["overview"] }); },
  });
}

export function useSetVerdict(callId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { verdict: T.Verdict; note: string }) => serialize(`decide:${callId}`, () => E.setVerdict(callId, body)),
    onSuccess: (summary) => {
      qc.setQueryData<T.ReviewPayload>(keys.call(callId), (prev) => (prev ? { ...prev, summary } : prev));
      qc.invalidateQueries({ queryKey: ["calls"] }); qc.invalidateQueries({ queryKey: ["overview"] });
    },
    onError: (e) => toast(`Could not save the verdict (${(e as Error).message}).`, 4000),
  });
}
