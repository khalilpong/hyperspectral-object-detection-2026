from pathlib import Path

from hsi_detection.annotations import (
    VocAnnotation,
    VocObject,
    read_voc_annotation,
    sanitize_annotation,
    to_yolo_rows,
)


def test_read_and_convert_voc(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_text(
        """<annotation>
        <filename>10.png</filename>
        <size><width>100</width><height>50</height><depth>16</depth></size>
        <object><name>rubik</name><bndbox>
        <xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>15</ymax>
        </bndbox></object></annotation>""",
        encoding="utf-8",
    )
    annotation = read_voc_annotation(xml)
    assert annotation.filename == "10.png"
    assert annotation.depth == 16
    assert to_yolo_rows(annotation, {"rubik": 15}) == [
        "15 0.20000000 0.20000000 0.20000000 0.20000000"
    ]


def test_sanitize_annotation_clips_and_drops_invalid_boxes() -> None:
    annotation = VocAnnotation(
        filename="sample.png",
        width=100,
        height=50,
        depth=16,
        objects=(
            VocObject("apple", 90, 10, 105, 20),
            VocObject("people", 100, 50, 100, 50),
        ),
    )
    result = sanitize_annotation(annotation)
    assert result.clipped_objects == 1
    assert result.dropped_objects == 1
    assert result.annotation.objects == (VocObject("apple", 90, 10, 100.0, 20),)
