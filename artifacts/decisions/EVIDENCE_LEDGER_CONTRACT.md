# EVIDENCE_LEDGER_CONTRACT

The runtime evidence ledger records policy-sensitive decisions with these fields:

1. timestamp (unix ms)
2. runtime mode (`strict` or `hardened`)
3. compatibility class (`known_compatible`, `known_incompatible`, `unknown`)
4. risk score
5. action (`allow`, `full_validate`, `fail_closed`)
6. note/context tag

This is the minimum schema required for decision auditability.
