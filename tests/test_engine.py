import unittest

from multimodal_rag.engine import (
    build_grounded_prompt,
    chunk_text,
    cosine_distance_to_similarity,
    document_id_for,
    RetrievedSource,
)


class ChunkTextTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(chunk_text(" \n\t "), [])

    def test_chunks_overlap_and_preserve_page_content(self) -> None:
        text = "one two three four five six seven eight"
        chunks = chunk_text(text, chunk_size=16, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks).replace("  ", " ").split()[0], "one")
        self.assertTrue(all(len(chunk) <= 16 for chunk in chunks))
        original_words = set(text.split())
        self.assertTrue(
            all(word in original_words for chunk in chunks for word in chunk.split())
        )
        self.assertEqual(original_words, set(" ".join(chunks).split()))
        self.assertTrue(set(chunks[0].split()) & set(chunks[1].split()))

    def test_invalid_chunk_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=10)

    def test_long_word_is_not_truncated_at_chunk_boundary(self) -> None:
        text = "an extraordinarilylongtoken continues here"
        chunks = chunk_text(text, chunk_size=10, overlap=2)

        self.assertTrue(
            any("extraordinarilylongtoken" in chunk for chunk in chunks)
        )
        self.assertIn("continues", " ".join(chunks))


class RetrievalHelperTests(unittest.TestCase):
    def test_cosine_distance_is_mapped_to_unit_similarity(self) -> None:
        self.assertEqual(cosine_distance_to_similarity(0), 1.0)
        self.assertEqual(cosine_distance_to_similarity(1), 0.5)
        self.assertEqual(cosine_distance_to_similarity(2), 0.0)
        self.assertEqual(cosine_distance_to_similarity(3), 0.0)

    def test_document_id_is_stable_for_same_file_bytes(self) -> None:
        self.assertEqual(document_id_for(b"sample"), document_id_for(b"sample"))
        self.assertNotEqual(document_id_for(b"sample"), document_id_for(b"other"))

    def test_grounded_prompt_includes_source_labels_and_page_text(self) -> None:
        source = RetrievedSource(
            source_id="S1",
            document_id="abc",
            filename="manual.pdf",
            page=3,
            text="The display shows battery status.",
            image_path="",
            text_score=0.9,
            image_score=0.8,
            score=0.86,
        )

        prompt = build_grounded_prompt("What does the display show?", [source])

        self.assertIn("[S1] manual.pdf, page 3", prompt)
        self.assertIn("The display shows battery status.", prompt)
        self.assertIn("What does the display show?", prompt)


if __name__ == "__main__":
    unittest.main()
