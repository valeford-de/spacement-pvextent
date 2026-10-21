# Additional quality exclusions after the two-check curation

The frozen V2 curation produced 247 positive and 238 negative partial masks
from the 1,000-photo collection; 515 cases remained completely ignored.
Its technical independent audit reproduced every new mask exactly and
confirmed the original metadata, geographic boundaries and inherited storage.
That proves the implementation follows its recorded approvals, not that the
approvals themselves are correct.

A separate deterministic visual spot check inspected twelve eligible positive
and twelve eligible negative cases, ordered by SHA-256 of sample ID. It read
native RGB and overlays of the actually saved masks. One positive case had
126 PV-labelled pixels whose boxes may cross panel frames and inter-row
substrate. Its entire supervision is withdrawn pending adjudication.

The spot check described the background regions of two other cases as
plausible while noting PV elsewhere in those photos. A metadata audit caught
that this contradicts the V2 requirement of explicit whole-image absence for
negative decisions. A subsequent native review of all twelve sampled negative
photos finds clear PV in one and unresolved glass/PV ambiguity in the other.
Both are also withdrawn, including their otherwise plausible BG patches.

The additional exclusions are:

| New sample | Reason |
| --- | --- |
| `nrw-bulk-77de82962f1b252603232bc0` | Uncertain PV interiors crossing frames or row bands |
| `nrw-bulk-ba3b9c4242f9c8e0d7888619` | Whole-image PV absence unresolved |
| `nrw-bulk-9d54fe305b920d71834cee4c` | Visible PV contradicts whole-image negative decision |

The first [quality protocol](nrw-bulk-mask-quality-quarantine-protocol-v1.json)
registered the positive exclusion. It was never executed. The subsequent
[V2 quality protocol](nrw-bulk-mask-quality-quarantine-protocol-v2.json) adds
the two contradictory negative cases before the first quarantine execution.
Both protocols, original reviews, initial consensus masks and prior snapshots
remain unchanged. No gate was lowered, no disputed pixel was relabelled as
background, and no model predictions or new training informed the decisions.

The separate quarantine stage creates `data/pvextent-training-v4`, keeping all
three photos available but moving them to unassigned with all-ignore masks.
It preserves every original base record and every other new record. The
original two-check V3 snapshot is historical evidence, not the final approved
training input.

These checks are experimental AI reviews. The small, selected spot check does
not estimate annotation error rates, certify all remaining labels, or provide
human benchmark ground truth. Shared proposals and correlated AI errors remain
limitations. Fuller target-source annotations and a separately reviewed
evaluation set are still required.
