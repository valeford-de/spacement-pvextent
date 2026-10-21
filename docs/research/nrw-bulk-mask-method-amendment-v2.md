# Shared-proposal mask checks: V2 method amendment

The user authorized two AI mask checks of all 1,000 additional NRW photos and
preparation of an immutable training dataset. The initial
[V1 protocol](nrw-bulk-mask-curation-protocol-v1.json) requested independently
drawn sparse module-interior and background polygons. No model training or
canonical bulk masks were produced under V1.

At the method checkpoint, 250 author-A drafts and 200 reviewer-B drawings had
been saved. Among 200 paired cases, 110 had two positive image decisions. Their
median PV intersection was four pixels; 49 had zero PV intersection. Only one
met the 100-PV-pixel gate, and none met both positive gates. Forty had two negative
decisions; six met the 1,024-background-pixel gate. These are workflow diagnostics,
not model performance or an annotation accuracy estimate.

An additional visual check of six paired positive cases found correctly placed
polygons on different module rows or parts of the same array. It found no
systematic coordinate transform error. Different sparse selections and arbitrary
background patch locations explained the lost intersections. Original drawings,
native inspection ledgers, quantitative diagnostics and visual QC remain outside
Git without revision.

The registered [V2 protocol](nrw-bulk-mask-curation-protocol-v2.json) changes the
method to **two independent checks of common AI visual proposals**. Former author
A continues preparing proposals for all 1,000 photos. Existing draft bytes are
preserved; new drafts are appended. Two new checkers C and D see the same frozen
proposal polygons and the corresponding original photos, without the author's
approval decisions/notes or the other checker's decisions. They explicitly
approve or reject each complete polygon. Uncertain or partly incorrect polygons
are rejected in full. No segmentation-model predictions or pseudo-ground-truth
are used.

Each batch's exact source and proposal hashes are fixed before its checks.
A combined proposal index is finalized after all draft batches are available.
The source queue, image hashes and every raw check are retained. Only polygons
approved by both checkers can become supervision. Positive cases still require
two positive decisions, at least 100 PV pixels and 256 background pixels;
negative cases require two explicit whole-image absence decisions, no approved
PV and at least 1,024 background pixels. Other pixels and deferred cases remain
ignore. Pixel gates, geographic protections and the prohibition on opening
reserved raster data are unchanged.

The resulting provenance must describe checks of shared proposals, rather than
two independently drawn masks. Both checkers can share a proposal's bias or
miss the same mistake; these are experimental AI-reviewed partial masks, not
human-confirmed benchmark truth. The V1 diagnostic and V2 method are separate,
and the change does not tune an evaluation threshold or a model result.
