# Phase A: Implement Workers, FusionManager, ResourceManager Improvements, Demo Runner, Scripts, Docs

Implements Phase‑A host prototype pieces:
- Stateless workers (fast/slow/raw) and FusionManager
- Authoritative ResourceManager + PlatformAllocator (memmap-backed)
- Thin MainLoop + ThreadingManager + CompletionDispatcher scaffolding
- Demo runner generating synthetic Bayer10 frames and exercising fast lane + fallback
- Scripts and docs for quick review and local testing

See ARCHITECTURE.md for detailed design and invariants.

### Checklist:
* Unit tests added? (follow-up)
* Telemetry wiring (follow-up)