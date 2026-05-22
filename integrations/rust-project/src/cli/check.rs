/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is dual-licensed under either the MIT license found in the
 * LICENSE-MIT file in the root directory of this source tree or the Apache
 * License, Version 2.0 found in the LICENSE-APACHE file in the root directory
 * of this source tree. You may select, at your option, one of the
 * above-listed licenses.
 */

use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use serde_json::json;

use crate::buck;
use crate::buck::Buck;
use crate::buck::BxlRecord;
use crate::buck::CheckStream;
use crate::cli::TargetOrFile;
use crate::diagnostics;

pub(crate) struct Check {
    pub(crate) buck: buck::Buck,
    pub(crate) use_clippy: bool,
    pub(crate) target_or_saved_file: TargetOrFile,
    /// Buck target patterns that should be checked on every save in addition
    /// to the saved file's owning target.
    pub(crate) always_check: Vec<String>,
}

impl Check {
    pub(crate) fn new(
        buck: Buck,
        use_clippy: bool,
        target_or_saved_file: TargetOrFile,
        always_check: Vec<String>,
    ) -> Self {
        let target_or_saved_file = target_or_saved_file.canonicalize();

        Self {
            buck,
            use_clippy,
            target_or_saved_file,
            always_check,
        }
    }

    #[tracing::instrument(name = "check", skip_all, fields(target = %self.target_or_saved_file))]
    pub(crate) fn run(&self) -> Result<(), anyhow::Error> {
        let start = std::time::Instant::now();
        let buck = &self.buck;

        let stream = match &self.target_or_saved_file {
            TargetOrFile::Target(target) => {
                buck.check_target(self.use_clippy, target, &self.always_check)?
            }
            TargetOrFile::File(saved_file) => {
                buck.check_saved_file(self.use_clippy, saved_file, &self.always_check)?
            }
        };

        // Lock stdout for the duration of the stream so partial diagnostic
        // lines from concurrent log output can't interleave with ours.
        let stdout = std::io::stdout();
        let mut writer = stdout.lock();
        stream_diagnostics(stream, &mut writer)?;

        crate::scuba::log_check(start.elapsed(), &self.target_or_saved_file, self.use_clippy);

        Ok(())
    }
}

/// Drive the BXL stream to completion, forwarding diagnostics to `writer` as
/// each target finishes.
///
/// For each target: read the diag_json file referenced by the streamed record,
/// rewrite span paths to be absolute, emit one `compiler-message` envelope per
/// rustc message, then a `compiler-artifact` sentinel for the target.
///
/// The `compiler-artifact` is what triggers per-package stale-diagnostic
/// clearing in rust-analyzer's flycheck — without it, targets that go from
/// "had errors" to "no errors" keep their stale entries on screen.
///
/// Note: the previous batch path deduplicated identical diagnostics across
/// targets. We do not — `package_id` attribution in the envelope is how
/// flycheck distinguishes per-target diagnostics, and dedup would defeat
/// per-target clearing when the same lib appears in multiple targets.
fn stream_diagnostics<W: Write>(
    mut stream: CheckStream,
    writer: &mut W,
) -> Result<(), anyhow::Error> {
    let mut project_root: Option<PathBuf> = None;

    while let Some(record) = stream.next_record()? {
        match record {
            BxlRecord::ProjectRoot {
                project_root: root,
            } => {
                project_root = Some(root);
            }
            BxlRecord::Diagnostic {
                diagnostic_path,
                target,
            } => {
                let root = project_root.as_deref().context(
                    "BXL emitted a diagnostic record before the project_root header",
                )?;
                forward_target(&diagnostic_path, &target, root, writer)?;
            }
        }
    }

    stream.finish()?;
    Ok(())
}

fn forward_target<W: Write>(
    diagnostic_path: &Path,
    target: &str,
    project_root: &Path,
    writer: &mut W,
) -> Result<(), anyhow::Error> {
    // `develop-json`'s `merge_unit_test_targets` (buck.rs) folds the generated
    // `*-unittest` test target into its parent lib in the produced
    // rust-project.json. The lib's `package_id` is what rust-analyzer indexes
    // by, so when the flycheck stream tags diagnostics with the unittest's own
    // label they get dropped on receipt. Strip the suffix here so unittest
    // diagnostics are attributed to the lib rust-analyzer actually knows about.
    let report_target = target.strip_suffix("-unittest").unwrap_or(target);

    let contents = match std::fs::read_to_string(diagnostic_path) {
        Ok(s) => s,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            // Buck claimed it materialized the artifact (otherwise `ensure`
            // would have failed in the BXL), but the file isn't on disk.
            // Happens when a target's compile fails before it reaches the
            // diag_json action — e.g. `:foo-unittest` depends on `:foo`,
            // and `:foo`'s rustc errored first. Skip and keep streaming
            // diagnostics for the targets that did produce them.
            tracing::warn!(
                target,
                path = %diagnostic_path.display(),
                "diagnostic JSON missing; target likely failed before emitting one"
            );
            return Ok(());
        }
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "reading diagnostic JSON for {target} at {}",
                    diagnostic_path.display(),
                )
            });
        }
    };

    for line in contents.lines() {
        // rustc emits one JSON message per line. File paths inside are
        // relative to the buck project root; promote them to absolute paths
        // so rust-analyzer can resolve them against its VFS regardless of
        // cwd.
        if let Ok(mut message) = serde_json::from_str::<diagnostics::Message>(line) {
            make_message_absolute(&mut message, project_root);
            let envelope = json!({
                "reason": "compiler-message",
                "package_id": report_target,
                "manifest_path": "",
                "target": cargo_target_stub(report_target),
                "message": message,
            });
            writeln!(writer, "{}", serde_json::to_string(&envelope)?)?;
        } else {
            // Forward unrecognised lines verbatim — rust-analyzer may
            // understand things we don't, and silently dropping would hide
            // information.
            writeln!(writer, "{line}")?;
        }
    }

    // Per-target sentinel: triggers `CheckMessage::CompilerArtifact` in
    // flycheck (crates/rust-analyzer/src/flycheck.rs), which is what
    // actually clears stale diagnostics for this target.
    let artifact = json!({
        "reason": "compiler-artifact",
        "package_id": report_target,
        "manifest_path": "",
        "target": cargo_target_stub(report_target),
        "profile": {
            "opt_level": "0",
            "debug_assertions": true,
            "overflow_checks": true,
            "test": false,
        },
        "features": [],
        "filenames": [],
        "executable": null,
        "fresh": false,
    });
    writeln!(writer, "{}", serde_json::to_string(&artifact)?)?;
    writer.flush()?;

    Ok(())
}

/// Minimal `cargo_metadata::Target` stub. flycheck only reads `name` and
/// `kind` for display; the other fields are required by the deserializer
/// but their values don't influence rust-analyzer behavior.
fn cargo_target_stub(target: &str) -> serde_json::Value {
    json!({
        "name": target,
        "kind": ["lib"],
        "crate_types": ["lib"],
        "required-features": [],
        "src_path": "",
        "edition": "2021",
        "doctest": false,
        "test": false,
        "doc": false,
    })
}

fn make_message_absolute(message: &mut diagnostics::Message, base_dir: &Path) {
    for span in message.spans.iter_mut() {
        make_span_absolute(span, base_dir);
    }

    for message in message.children.iter_mut() {
        make_message_absolute(message, base_dir);
    }
}

fn make_span_absolute(span: &mut diagnostics::Span, base_dir: &Path) {
    span.file_name = base_dir.join(&span.file_name);

    if let Some(expansion) = &mut span.expansion {
        if let Some(def_site_span) = &mut expansion.def_site_span {
            make_span_absolute(def_site_span, base_dir);
        }

        make_span_absolute(&mut expansion.span, base_dir);
    }
}
