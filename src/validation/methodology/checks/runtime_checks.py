"""Runtime methodology checks evaluated from actual data (prepared, not wired yet).

Runtime checks are registered exactly like static ones, only with
``phase=RulePhase.RUNTIME``; their ``CheckContext.data`` carries the runtime
object (e.g. ``DatasetSplitDTO`` right after the splitting stage). They are
evaluated by ``MethodologyValidator.validate_runtime``.

Planned checks (see the commented-out rules in ``configs/methodology/rules.yaml``):
    - ``subjects_disjoint`` (PROT-002, scenario D): no subject id in both train and test.
    - ``sessions_disjoint``: no session in both train and test for cross-session.
    - ``trials_disjoint`` (GRP-001, scenario C): no (subject, session, run, trial) in both train and test.
    - ``test_set_access_count`` (SEL-003, scenario B): test data is used only for the final evaluation.

Prerequisite: every epoch in ``EpochPreprocessedDTO`` must keep its provenance
(subject, session, run, trial, time interval).

Example::

    @methodology_check("subjects_disjoint", phase=RulePhase.RUNTIME)
    def subjects_disjoint(ctx: CheckContext) -> list[Finding]:
        split: DatasetSplitDTO = ctx.data
        ...
"""
