# Variable Dictionary

Common fields used in the released aggregate and identifier CSV files:

- `dataset`: released dataset/site label, such as `ngsim_us101`, `ngsim_i80`, or `highd_external`.
- `membership`: processed set membership, e.g. `modeled_crash_denominator` or `noncrash_complement`.
- `base_scenario_id`: processed base episode identifier before physical uncertainty replication.
- `scenario_id`: processed replicated scenario identifier.
- `source`: standardized source family, such as `ngsim` or `highd`.
- `family`: rear-end scenario family label.
- `ego_vehicle_id`, `lead_vehicle_id`: source vehicle identifiers retained only as processed IDs.
- `episode_id`: processed episode identifier within a site/source file.
- `replicate_id`: uncertainty-replication index.
- `recording_id`, `location_id`, `driving_direction`: highD-derived recording/stratum identifiers.
- `delta_s`: available lead time grid value in seconds.
- `coverage_*`: coverage curve value for a policy/controller.
- `saturation`: asymptotic coverage value reported in aggregate tables.
- `delta_abs_q90_s`, `delta_rel_q90_s`: reported q90 lead-time quantities.
- `sat_gap_vs_star`: residual saturation gap relative to the upper-bound/reference policy.
- `ci_low`, `ci_high`: bootstrap interval bounds where available.
