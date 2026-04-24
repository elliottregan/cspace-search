//! Text chunker with line-number tracking. Ports
//! `search/corpus/code_chunker.go` byte-for-byte so embeddings built by
//! the Go version continue to index at equivalent offsets.

#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Maximum characters per chunk. When `<= 0`, falls back to 12 000.
    pub max: usize,
    /// Character overlap between consecutive chunks.
    pub overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max: 12_000,
            overlap: 200,
        }
    }
}

/// One contiguous text slice with a 1-based inclusive line range.
#[derive(Debug, Clone)]
pub struct ChunkOut {
    pub text: String,
    pub line_start: u32,
    pub line_end: u32,
}

/// Split `content` into chunks respecting max size and overlap. Line
/// numbers are 1-based inclusive.
pub fn chunk(content: &[u8], cfg: ChunkConfig) -> Vec<ChunkOut> {
    let s = String::from_utf8_lossy(content).into_owned();
    let max = if cfg.max == 0 { 12_000 } else { cfg.max };

    if s.len() <= max {
        return vec![ChunkOut {
            text: s.clone(),
            line_start: 1,
            line_end: line_count(&s),
        }];
    }

    let mut out: Vec<ChunkOut> = Vec::new();
    let mut start = 0usize;
    while start < s.len() {
        let mut end = start + max;
        if end >= s.len() {
            end = s.len();
        } else {
            // Prefer to cut at a newline within the last 500 chars of the
            // max window to avoid mid-line splits.
            let window = &s[start..end];
            if let Some(nl) = window.rfind('\n') {
                if nl > max.saturating_sub(500) && nl > 0 {
                    end = start + nl + 1;
                }
            }
        }
        // Ensure char boundary (Go sliced bytes freely; Rust must be safe).
        while end < s.len() && !s.is_char_boundary(end) {
            end += 1;
        }
        while start > 0 && !s.is_char_boundary(start) {
            start += 1;
        }

        let text = s[start..end].to_string();
        let ls = line_of(&s, start);
        let le = line_of(&s, end.saturating_sub(1));
        out.push(ChunkOut {
            text,
            line_start: ls,
            line_end: le,
        });

        if end >= s.len() {
            break;
        }
        let next = end.saturating_sub(cfg.overlap);
        start = if next <= start { start + 1 } else { next };
    }
    out
}

fn line_count(s: &str) -> u32 {
    if s.is_empty() {
        return 1;
    }
    let mut n = s.matches('\n').count() as u32;
    if !s.ends_with('\n') {
        n += 1;
    }
    if n == 0 {
        n = 1;
    }
    n
}

fn line_of(s: &str, off: usize) -> u32 {
    if s.is_empty() {
        return 1;
    }
    let off = off.min(s.len().saturating_sub(1));
    1 + s[..=off].matches('\n').count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_file_one_chunk_whole_file() {
        let content = "line1\nline2\nline3\n";
        let chunks = chunk(
            content.as_bytes(),
            ChunkConfig {
                max: 12_000,
                overlap: 0,
            },
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 3);
        assert_eq!(chunks[0].text, content);
    }

    #[test]
    fn empty_file_one_empty_chunk() {
        let chunks = chunk(
            b"",
            ChunkConfig {
                max: 100,
                overlap: 0,
            },
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 1);
    }

    #[test]
    fn large_file_splits_with_overlap() {
        let mut b = String::new();
        for _ in 0..200 {
            b.push_str(&"x".repeat(99));
            b.push('\n');
        }
        let chunks = chunk(
            b.as_bytes(),
            ChunkConfig {
                max: 8_000,
                overlap: 200,
            },
        );
        assert!(chunks.len() >= 2, "expected ≥2 chunks, got {}", chunks.len());
        assert_eq!(chunks[0].line_start, 1);
        for i in 1..chunks.len() {
            assert!(
                chunks[i].line_start <= chunks[i - 1].line_end,
                "chunk {i} starts at line {} after previous ended at {} — no overlap",
                chunks[i].line_start,
                chunks[i - 1].line_end
            );
        }
        assert!(
            chunks.last().unwrap().line_end >= 200,
            "last chunk ends at {}, expected ≥200",
            chunks.last().unwrap().line_end
        );
    }

    #[test]
    fn no_trailing_newline() {
        let chunks = chunk(
            b"one\ntwo\nthree",
            ChunkConfig {
                max: 100,
                overlap: 0,
            },
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_end, 3);
    }
}
