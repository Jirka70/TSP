"""Static checks of the preprocessing configuration."""

from __future__ import annotations

from src.validation.methodology.check_registry import CheckContext, Finding, methodology_check


@methodology_check("effective_band_matches_paradigm_filter")
def effective_band_matches_paradigm_filter(check_context: CheckContext) -> list[Finding]:
    """Raw filters and resampling must not narrow the band declared in ``paradigm.filter``.

    Resampling to a sampling frequency removes everything above its Nyquist frequency
    (sampling frequency / 2), so the declared upper edge must stay below it.
    """
    raw_preprocessing = check_context.config.raw_preprocessing
    paradigm = check_context.config.paradigm
    declared_low_frequency = paradigm.filter.fmin
    declared_high_frequency = paradigm.filter.fmax
    declared_band = f"{declared_low_frequency}-{declared_high_frequency} Hz"
    findings: list[Finding] = []

    high_pass_filter = raw_preprocessing.high_pass_filter
    if high_pass_filter.enabled and high_pass_filter.l_freq > declared_low_frequency:
        findings.append(
            Finding(
                detail=f"Raw high-pass at {high_pass_filter.l_freq} Hz removes the lower part of the declared band "
                       f"{declared_band}.",
                location="raw_preprocessing.high_pass_filter.l_freq",
            )
        )

    low_pass_filter = raw_preprocessing.low_pass_filter
    if low_pass_filter.enabled and low_pass_filter.h_freq < declared_high_frequency:
        findings.append(
            Finding(
                detail=f"Raw low-pass at {low_pass_filter.h_freq} Hz removes the upper part of the declared band "
                       f"{declared_band}.",
                location="raw_preprocessing.low_pass_filter.h_freq",
            )
        )

    raw_resampling = raw_preprocessing.resampling
    if raw_resampling.enabled and raw_resampling.sfreq / 2 <= declared_high_frequency:
        findings.append(
            Finding(
                detail=f"Raw resampling to {raw_resampling.sfreq} Hz keeps only frequencies below "
                       f"{raw_resampling.sfreq / 2} Hz, but the declared band is {declared_band}.",
                location="raw_preprocessing.resampling.sfreq",
            )
        )

    epoch_resampling = paradigm.resampling
    if epoch_resampling.enabled and epoch_resampling.sfreq / 2 <= declared_high_frequency:
        findings.append(
            Finding(
                detail=f"Epoch resampling to {epoch_resampling.sfreq} Hz keeps only frequencies below "
                       f"{epoch_resampling.sfreq / 2} Hz, but the declared band is {declared_band}.",
                location="paradigm.resampling.sfreq",
            )
        )

    return findings
