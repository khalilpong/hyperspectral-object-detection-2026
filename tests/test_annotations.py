from pathlib import Path

from hsi_detection.annotations import read_voc_annotation, to_yolo_rows


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

