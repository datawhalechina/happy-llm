import unittest

from utils import ReadFiles, enc


class GetChunkTest(unittest.TestCase):
    def test_long_chinese_line_uses_token_overlap(self):
        max_token_len = 20
        cover_content = 5

        chunks = ReadFiles.get_chunk(
            "你好世界" * 30,
            max_token_len=max_token_len,
            cover_content=cover_content,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(len(enc.encode(chunk)) <= max_token_len for chunk in chunks)
        )
        for previous, current in zip(chunks, chunks[1:]):
            expected_overlap = enc.decode(enc.encode(previous)[-cover_content:])
            self.assertTrue(current.startswith(expected_overlap))

    def test_unicode_character_is_not_split_across_token_boundaries(self):
        text = "😊"
        self.assertEqual(2, len(enc.encode(text)))

        chunks = ReadFiles.get_chunk(
            text,
            max_token_len=2,
            cover_content=1,
        )

        self.assertEqual([text], chunks)
        self.assertNotIn("\ufffd", "".join(chunks))
        self.assertTrue(all(len(enc.encode(chunk)) <= 2 for chunk in chunks))

    def test_overlap_does_not_consume_new_content_budget_twice(self):
        chunks = ReadFiles.get_chunk(
            "\n".join(["alpha"] * 4),
            max_token_len=6,
            cover_content=2,
        )

        self.assertEqual(2, len(chunks))
        self.assertTrue(all(len(enc.encode(chunk)) <= 6 for chunk in chunks))
        self.assertEqual(["alpha", "alpha"], chunks[-1].splitlines()[-2:])

    def test_overlap_is_kept_before_first_part_of_long_line(self):
        chunks = ReadFiles.get_chunk(
            "alpha\n" + "你好世界" * 20,
            max_token_len=10,
            cover_content=2,
        )

        self.assertEqual("alpha", chunks[0])
        self.assertTrue(chunks[1].startswith("alpha\n"))
        self.assertTrue(all(len(enc.encode(chunk)) <= 10 for chunk in chunks))

    def test_rejects_invalid_token_budgets(self):
        invalid_arguments = (
            {"max_token_len": 0, "cover_content": 0},
            {"max_token_len": 10, "cover_content": -1},
            {"max_token_len": 10, "cover_content": 10},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    ReadFiles.get_chunk("text", **arguments)


if __name__ == "__main__":
    unittest.main()
