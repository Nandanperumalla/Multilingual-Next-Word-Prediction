"""Build the research presentation, speaker notes and outline.

Run with a Python that has python-pptx installed, from anywhere:
    python build_presentation.py

Metrics are read from outputs/*.json at build time. English results are shown as
"In progress" until outputs/english_families/comparison.json exists; the script warns when it does.
Editing the .pptx by hand is fine, but rebuilding replaces those edits.
"""
import copy
import json
import re
from datetime import datetime
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs'

NAVY, TEAL, BLUE, INK, MUTED = '14284B', '0F766E', '5B7FA6', '1F2937', '5F6B7A'
LIGHT, PALE_TEAL, PALE_BLUE, LINE, WHITE = 'F3F5F8', 'E3F2F0', 'E6EDF5', 'CBD3DD', 'FFFFFF'
AMBER, PALE_AMBER = '92400E', 'FEF3DC'
FONT = 'Arial'
FOOTER = 'Multilingual Next-Word Prediction  |  UROP Final Evaluation'
LEFT, RIGHT, WIDTH = 0.6, 12.73, 12.13
IN_PROGRESS = 'In progress'


def rgb(value):
    return RGBColor.from_string(value)


# ---------------------------------------------------------------- data


def read(path):
    return json.loads((OUT / path).read_text(encoding='utf-8'))


def pct(value):
    return f'{value * 100:.2f}'


def history(path, key='top1_accuracy'):
    return [row['validation'][key] * 100 for row in read(path)]


TE = {
    'lstm': read('telugu_comparison/lstm_test_metrics.json'),
    'bilstm': read('telugu_comparison/bilstm_test_metrics.json'),
    'cnn': read('telugu_cnn/cnn_test_metrics.json'),
    'wordchar': read('telugu_wordchar/wordchar_cnn_test_metrics.json'),
    'v2': read('telugu_accuracy_v2/bilstm_test_metrics.json'),
    'v3': read('telugu_accuracy_v3/bilstm_test_metrics.json'),
}
TE_CNN2 = read('telugu_cnn_accuracy_v2/cnn_test_metrics.json')
TE_H1 = read('telugu_hybrid_v1/test_metrics.json')
TE_ENS = read('telugu_ensemble/test_metrics.json')
HI = {
    'transformer': read('hindi_families/transformer_test_metrics.json'),
    'bilstm': read('hindi_comparison/bilstm_test_metrics.json'),
    'wordchar': read('hindi_families/wordchar_cnn_test_metrics.json'),
    'cnn': read('hindi_families/cnn_test_metrics.json'),
    'lstm': read('hindi_comparison/lstm_test_metrics.json'),
    'gru': read('hindi_families/gru_test_metrics.json'),
}
SEEDS = read('telugu_seeds/summary.json')
EN_PROVENANCE = json.loads((ROOT / 'data_english/provenance.json').read_text(encoding='utf-8'))
HI_ROWS = {row['key']: row for row in read('hindi_families/comparison.json')['models']}
HI_COMBINED, HI_HYBRID = HI_ROWS['ensemble'], HI_ROWS['hybrid']
HI_WEIGHTS = read('hindi_families/comparison.json')['weights']
if (OUT / 'english_families/comparison.json').exists():
    print('WARNING: english comparison.json now exists. Update the English text and tables by hand.')

# ---------------------------------------------------------------- references (numbered by first use)

REFERENCES = {
    'hard': 'A. Hard et al., “Federated learning for mobile keyboard prediction,” arXiv:1811.03604, 2018.',
    'srinagesh': 'A. Srinagesh, Ch. Sudha Sree, B. Prasanthi, P. Rama Krishna, and P. Siva Prasad, “Next word prediction in Telugu sentences using recurrent neural networks,” Turkish Online J. Qualitative Inquiry, vol. 12, no. 5, pp. 4284–4292, 2021.',
    'saraswathi': 'R. Vijaya Saraswathi et al., “Next word prediction in Telugu using RNN mechanism,” in Proc. Int. Conf. Augmented Intelligence and Sustainable Systems (ICAISS), 2022, doi: 10.1109/ICAISS55157.2022.10010963.',
    'sharma': 'R. Sharma, N. Goel, N. Aggarwal, P. Kaur, and C. Prakash, “Next word prediction in Hindi using deep learning techniques,” in Proc. Int. Conf. Data Science and Engineering (ICDSE), 2019, pp. 55–60, doi: 10.1109/ICDSE47409.2019.8971796.',
    'lstm': 'S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, 1997.',
    'bojanowski': 'P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, “Enriching word vectors with subword information,” arXiv:1607.04606, 2016.',
    'barman': 'P. P. Barman and A. Boruah, “A RNN based approach for next word prediction in Assamese phonetic transcription,” Procedia Comput. Sci., vol. 143, pp. 117–123, 2018, doi: 10.1016/j.procs.2018.10.359.',
    'endalie': 'D. Endalie, G. Haile, and W. Taye, “Bi-directional long short term memory-gated recurrent unit model for Amharic next word prediction,” PLoS ONE, vol. 17, no. 8, e0273156, 2022, doi: 10.1371/journal.pone.0273156.',
    'islam': 'M. R. Islam, A. Amin, and A. N. Zereen, “Enhancing Bangla language next word prediction and sentence completion through extended RNN with Bi-LSTM model on N-gram language,” arXiv:2405.01873, 2024.',
    'chhetri': 'P. Chhetri, T. Yoezer, P. Wangmo, and T. Bomjan, “Dzongkha next word prediction system,” arXiv:2607.11515, 2026.',
    'lora': 'E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” arXiv:2106.09685, 2021.',
    'bloom': 'BigScience Workshop, “BLOOM: A 176B-parameter open-access multilingual language model,” arXiv:2211.05100, 2022. Smaller model used: huggingface.co/bigscience/bloom-560m.',
    'bilstm': 'M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Trans. Signal Process., vol. 45, no. 11, pp. 2673–2681, 1997.',
    'gru': 'K. Cho et al., “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” in Proc. EMNLP, 2014, pp. 1724–1734.',
    'transformer': 'A. Vaswani et al., “Attention is all you need,” arXiv:1706.03762, 2017.',
    'te_alpaca': 'Telugu-LLM-Labs, “telugu_alpaca_yahma_cleaned_filtered_romanized,” Hugging Face dataset. [Online]. Available: huggingface.co/datasets/Telugu-LLM-Labs',
    'te_gpteacher': 'Telugu-LLM-Labs, “telugu_teknium_GPTeacher_general_instruct_filtered_romanized,” Hugging Face dataset. [Online]. Available: huggingface.co/datasets/Telugu-LLM-Labs',
    'hi_alpaca': 'iamshnoo, “alpaca-cleaned-hindi,” Hugging Face dataset. [Online]. Available: huggingface.co/datasets/iamshnoo/alpaca-cleaned-hindi',
    'en_alpaca': 'yahma, “alpaca-cleaned,” Hugging Face dataset. [Online]. Available: huggingface.co/datasets/yahma/alpaca-cleaned',
    'nllb': 'NLLB Team (M. R. Costa-jussà et al.), “No Language Left Behind: Scaling human-centered machine translation,” arXiv:2207.04672, 2022.',
    'adam': 'D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. ICLR, 2015, arXiv:1412.6980.',
}
cited = []


def c(*keys):
    """Return an IEEE citation, numbering each reference at its first use."""
    numbers = []
    for key in keys:
        if key not in cited:
            cited.append(key)
        numbers.append(cited.index(key) + 1)
    return ', '.join(f'[{n}]' for n in sorted(numbers))


# ---------------------------------------------------------------- presentation setup

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
for index, layout in reversed(list(enumerate(prs.slide_layouts))):
    if index not in (0, 5):
        prs.slide_layouts.remove(layout)
TITLE_LAYOUT, CONTENT_LAYOUT = prs.slide_layouts[0], prs.slide_layouts[1]
master = prs.slide_master


def set_theme():
    part = master.part.part_related_by(RT.THEME)
    xml = part.blob.decode('utf-8')
    xml = re.sub(r'(<a:(?:major|minor)Font>\s*<a:latin typeface=")[^"]*', r'\g<1>' + FONT, xml)
    scheme = dict(dk2=NAVY, lt2=LIGHT, accent1=NAVY, accent2=TEAL, accent3=BLUE, accent4=AMBER, accent5=MUTED, accent6=LINE)
    for name, value in scheme.items():
        xml = re.sub(rf'(<a:{name}>)\s*<a:[a-zA-Z]+Clr [^>]*/>\s*(</a:{name}>)', rf'\g<1><a:srgbClr val="{value}"/>\g<2>', xml)
    part._blob = xml.encode('utf-8')


def place(shape, x, y, w, h):
    shape.left, shape.top, shape.width, shape.height = Inches(x), Inches(y), Inches(w), Inches(h)


def set_master():
    set_theme()
    for shape in list(master.placeholders):
        kind = shape.placeholder_format.type
        if 'TITLE' in str(kind):
            place(shape, LEFT, 0.36, WIDTH, 0.8)
            shape._element.find(qn('p:txBody')).find(qn('a:bodyPr')).set('lIns', '0')
        elif 'BODY' in str(kind):
            place(shape, LEFT, 1.35, WIDTH, 5.3)
        elif 'FOOTER' in str(kind):
            place(shape, LEFT, 6.96, 9.0, 0.3)
            shape._element.find(qn('p:txBody')).find(qn('a:bodyPr')).set('lIns', '0')
        elif 'SLIDE_NUMBER' in str(kind):
            place(shape, 11.53, 6.96, 1.2, 0.3)
            shape._element.find(qn('p:txBody')).find(qn('a:bodyPr')).set('rIns', '0')
        elif 'DATE' in str(kind):
            shape._element.getparent().remove(shape._element)
    title_style = master._element.find(qn('p:txStyles')).find(qn('p:titleStyle')).find(qn('a:lvl1pPr'))
    title_style.set('algn', 'l')
    title_run = title_style.find(qn('a:defRPr'))
    title_run.set('sz', '3000')
    title_run.set('b', '1')
    fill = title_run.find(qn('a:solidFill'))
    fill.clear()
    etree.SubElement(fill, qn('a:srgbClr')).set('val', NAVY)
    for layout in prs.slide_layouts:
        for shape in list(layout.placeholders):
            kind = str(shape.placeholder_format.type)
            if 'DATE' in kind:
                shape._element.getparent().remove(shape._element)
            elif 'FOOTER' in kind or 'SLIDE_NUMBER' in kind or kind.startswith('TITLE') and layout is CONTENT_LAYOUT:
                sp_pr = shape._element.spPr
                for child in list(sp_pr):
                    sp_pr.remove(child)


set_master()

SLIDES = []


def set_footer(slide, number):
    for shape in slide.slide_layout.placeholders:
        kind = str(shape.placeholder_format.type)
        if 'FOOTER' not in kind and 'SLIDE_NUMBER' not in kind:
            continue
        element = copy.deepcopy(shape._element)
        slide.shapes._spTree.append(element)
        body = element.find(qn('p:txBody'))
        paragraph = body.findall(qn('a:p'))[0]
        for child in list(paragraph):
            if child.tag != qn('a:pPr'):
                paragraph.remove(child)
        p_pr = paragraph.find(qn('a:pPr'))
        if p_pr is None:
            p_pr = etree.SubElement(paragraph, qn('a:pPr'))
        p_pr.set('algn', 'r' if 'SLIDE_NUMBER' in kind else 'l')
        attributes = dict(lang='en-GB', sz='1100')
        if 'FOOTER' in kind:
            run = etree.SubElement(paragraph, qn('a:r'))
            r_pr = etree.SubElement(run, qn('a:rPr'), attributes)
            etree.SubElement(etree.SubElement(r_pr, qn('a:solidFill')), qn('a:srgbClr')).set('val', MUTED)
            etree.SubElement(run, qn('a:t')).text = FOOTER
        else:
            field = etree.SubElement(paragraph, qn('a:fld'), id='{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}', type='slidenum')
            r_pr = etree.SubElement(field, qn('a:rPr'), attributes)
            etree.SubElement(etree.SubElement(r_pr, qn('a:solidFill')), qn('a:srgbClr')).set('val', MUTED)
            etree.SubElement(field, qn('a:t')).text = str(number)


def new_slide(title, notes, visuals, title_slide=False):
    slide = prs.slides.add_slide(TITLE_LAYOUT if title_slide else CONTENT_LAYOUT)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(WHITE)
    slide.shapes.title.text = title
    number = len(prs.slides)
    if not title_slide:
        set_footer(slide, number)
    slide.notes_slide.notes_text_frame.text = notes
    SLIDES.append(dict(number=number, title=title, notes=notes, visuals=visuals))
    return slide


# ---------------------------------------------------------------- drawing helpers

TOKEN = re.compile(r'(\*\*.+?\*\*|_\{.+?\}|\^\{.+?\}|\{\{.+?\}\})')


def add_runs(paragraph, text, size, color, bold=False, font=FONT):
    for part in TOKEN.split(text):
        if not part:
            continue
        run_color, run_bold, baseline = color, bold, None
        if part.startswith('**'):
            part, run_bold = part[2:-2], True
        elif part.startswith('_{'):
            part, baseline = part[2:-1], '-25000'
        elif part.startswith('^{'):
            part, baseline = part[2:-1], '30000'
        elif part.startswith('{{'):
            part, run_color = part[2:-2], AMBER
        run = paragraph.add_run()
        run.text = part
        run.font.size, run.font.bold, run.font.name = Pt(size), run_bold, font
        run.font.color.rgb = rgb(run_color)
        if baseline:
            run._r.get_or_add_rPr().set('baseline', baseline)


def set_bullet(paragraph, size):
    p_pr = paragraph._p.get_or_add_pPr()
    indent = int(Pt(size).emu * 0.95)
    p_pr.set('marL', str(indent))
    p_pr.set('indent', str(-indent))
    color = etree.SubElement(p_pr, qn('a:buClr'))
    etree.SubElement(color, qn('a:srgbClr')).set('val', TEAL)
    etree.SubElement(p_pr, qn('a:buFont')).set('typeface', 'Arial')
    etree.SubElement(p_pr, qn('a:buChar')).set('char', '•')


def fill_text(frame, lines, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, bullets=False, gap=6, anchor=MSO_ANCHOR.TOP,
              inset=(0, 0, 0, 0), spacing=None):
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left, frame.margin_top, frame.margin_right, frame.margin_bottom = [Inches(v) for v in inset]
    lines = [lines] if isinstance(lines, str) else lines
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        if index:
            paragraph.space_before = Pt(gap)
        if spacing:
            paragraph.line_spacing = spacing
        if bullets:
            set_bullet(paragraph, size)
        add_runs(paragraph, line, size, color, bold)


def text(slide, x, y, w, h, lines, name=None, **options):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    fill_text(box.text_frame, lines, **options)
    if name:
        box.name = name
    return box


def box(slide, x, y, w, h, lines='', fill=LIGHT, line=None, size=15, color=INK, bold=False, align=PP_ALIGN.CENTER,
        anchor=MSO_ANCHOR.MIDDLE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, inset=(0.1, 0.05, 0.1, 0.05), name=None, **options):
    item = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        item.adjustments[0] = min(0.08, 0.08 / max(min(w, h), 0.01))
    item.shadow.inherit = False
    item.fill.solid()
    item.fill.fore_color.rgb = rgb(fill)
    if line:
        item.line.color.rgb = rgb(line)
        item.line.width = Pt(1)
    else:
        item.line.fill.background()
    fill_text(item.text_frame, lines, size=size, color=color, bold=bold, align=align, anchor=anchor, inset=inset, **options)
    if name:
        item.name = name
    return item


def arrow(slide, start, end, start_site=3, end_site=1, color=BLUE):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 0, 0, 0, 0)
    line.begin_connect(start, start_site)
    line.end_connect(end, end_site)
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(1.75)
    outline = line.line._get_or_add_ln()
    etree.SubElement(outline, qn('a:tailEnd'), type='triangle', w='med', len='med')
    line.name = 'Arrow'
    return line


def cell_border(cell, color=LINE):
    tc_pr = cell._tc.get_or_add_tcPr()
    for tag in ('a:lnB', 'a:lnT', 'a:lnR', 'a:lnL'):
        if tag in ('a:lnL', 'a:lnR'):
            line = etree.Element(qn(tag), w='0')
            etree.SubElement(line, qn('a:noFill'))
        else:
            line = etree.Element(qn(tag), w='9525', cap='flat', cmpd='sng', algn='ctr')
            etree.SubElement(etree.SubElement(line, qn('a:solidFill')), qn('a:srgbClr')).set('val', color)
            etree.SubElement(line, qn('a:prstDash')).set('val', 'solid')
        tc_pr.insert(0, line)


def table(slide, x, y, widths, rows, row_height=0.4, size=14, header_size=None, aligns=None, highlight=(), muted=(), name='Table', header_height=None):
    """Native table. Rows are lists of strings; the first row is the header."""
    frame = slide.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_height * len(rows)))
    frame.name = name
    grid = frame.table
    grid_pr = grid._tbl.tblPr
    grid_pr.set('bandRow', '0')
    grid_pr.set('firstRow', '1')
    for index, width in enumerate(widths):
        grid.columns[index].width = Inches(width)
    for r, row in enumerate(rows):
        grid.rows[r].height = Inches(header_height if header_height and r == 0 else row_height)
        for k, value in enumerate(row):
            cell = grid.cell(r, k)
            cell.margin_left, cell.margin_right = Inches(0.08), Inches(0.08)
            cell.margin_top, cell.margin_bottom = Inches(0.03), Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            header = r == 0
            cell.fill.fore_color.rgb = rgb(NAVY if header else PALE_TEAL if r in highlight else WHITE)
            colour = WHITE if header else MUTED if r in muted else INK
            if value == IN_PROGRESS:
                cell.fill.fore_color.rgb = rgb(PALE_AMBER)
                colour = AMBER
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = (aligns[k] if aligns else PP_ALIGN.LEFT)
            add_runs(paragraph, value, header_size or size, colour, bold=header or r in highlight)
            cell_border(cell)
    return frame


def describe(shape, description):
    shape._element.xpath('.//p:cNvPr')[0].set('descr', description)


def style_chart(chart, size=13, legend=True):
    chart.font.size, chart.font.name = Pt(size), FONT
    chart.font.color.rgb = rgb(INK)
    chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
    value_axis = chart.value_axis
    value_axis.major_gridlines.format.line.color.rgb = rgb('E5E9EF')
    value_axis.format.line.fill.background()
    value_axis.tick_labels.font.size = Pt(size)
    chart.category_axis.format.line.color.rgb = rgb(LINE)
    chart.category_axis.tick_labels.font.size = Pt(size)


def axis_title(axis, label, size=13):
    axis.has_title = True
    axis.axis_title.text_frame.text = label
    run = axis.axis_title.text_frame.paragraphs[0].runs[0]
    run.font.size, run.font.bold = Pt(size), False
    run.font.color.rgb = rgb(MUTED)


def line_chart(slide, x, y, w, h, categories, series, colors, minimum, maximum, unit, description):
    data = CategoryChartData()
    data.categories = categories
    for label, values in series:
        data.add_series(label, values)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), data)
    frame.name = 'Chart: ' + description
    describe(frame, description)
    chart = frame.chart
    style_chart(chart, 12)
    axis = chart.value_axis
    axis.minimum_scale, axis.maximum_scale, axis.major_unit = minimum, maximum, unit
    axis.tick_labels.number_format, axis.tick_labels.number_format_is_linked = '0', False
    axis_title(axis, 'Validation Top-1 (%)', 12)
    axis_title(chart.category_axis, 'Epoch', 12)
    for plot_series, color in zip(chart.plots[0].series, colors):
        plot_series.smooth = False
        plot_series.format.line.color.rgb = rgb(color)
        plot_series.format.line.width = Pt(2.25)
        plot_series.marker.style = XL_MARKER_STYLE.CIRCLE
        plot_series.marker.size = 6
        plot_series.marker.format.fill.solid()
        plot_series.marker.format.fill.fore_color.rgb = rgb(color)
        plot_series.marker.format.line.color.rgb = rgb(color)
    return frame


C, R = PP_ALIGN.CENTER, PP_ALIGN.RIGHT
L = PP_ALIGN.LEFT

# ---------------------------------------------------------------- 1 Title

slide = new_slide(
    'Multilingual Next-Word Prediction',
    'Good morning. This project compares next-word prediction models for Telugu, Hindi and English under one evaluation '
    'method. Telugu and Hindi are complete apart from the Hindi pretrained test. English models are still training.',
    ['University logo taken from the report cover', 'Amber fields mark details the student must supply'], title_slide=True)
title_shape = slide.shapes.title
place(title_shape, LEFT, 1.0, 9.6, 1.9)
title_shape.text_frame.word_wrap = True
title_shape.text_frame.vertical_anchor = MSO_ANCHOR.TOP
for paragraph in title_shape.text_frame.paragraphs:
    paragraph.alignment = L
    for run in paragraph.runs:
        run.font.size, run.font.bold = Pt(36), True
        run.font.color.rgb = rgb(NAVY)
subtitle = slide.placeholders[1]
place(subtitle, LEFT, 3.0, 9.6, 0.85)
subtitle.text_frame.margin_left = 0
subtitle.text_frame.clear()
fill_text(subtitle.text_frame, 'A comparative study of recurrent, convolutional, Transformer, count-based and combined models',
          size=20, color=TEAL)
logo = slide.shapes.add_picture(str(HERE / 'assets/srm_ap_logo.png'), Inches(10.93), Inches(1.0), Inches(1.8), Inches(1.8))
logo.name = 'SRM University-AP logo'
describe(logo, 'SRM University-AP logo from the project report cover')
detail_rows = [
    ('Presented by', 'A. Spurthi (AP23110011045), P. Sathvik (AP23110010752), P. Nandan (AP23110010918)'),
    ('Supervisor', 'Dr. Boyapati Prasanthi'),
    ('Department', 'Computer Science and Engineering, School of Engineering and Sciences'),
    ('University', 'SRM University–AP, Andhra Pradesh'),
    ('Session', 'UROP Project, B.Tech Final Evaluation  |  {{[Academic year: required]}}  |  November 2026'),
]
for index, (label, value) in enumerate(detail_rows):
    y = 4.2 + index * 0.46
    text(slide, LEFT, y, 1.9, 0.4, label, size=16, color=MUTED, name=f'{label} label')
    text(slide, 2.5, y, 10.2, 0.4, value, size=16, name=f'{label} value')

# ---------------------------------------------------------------- 2 Abstract

ABSTRACT = [
    ('Background', 'Next-word prediction reduces typing effort, yet published work concentrates on English and test methods are often unclear.'),
    ('Method', 'Recurrent, convolutional, Transformer, count-based and combined models share one method: grouped 90/5/5 splits, removal of exact sentence overlap, a 20,000-word training-only vocabulary and an eight-word context.'),
    ('Results', f'On 200,092 unseen Telugu examples, the combined model reached {pct(TE_ENS["ensemble"]["top1_accuracy"])}% Top-1 and {pct(TE_ENS["ensemble"]["top5_accuracy"])}% Top-5. On 296,193 machine-translated Hindi examples, the combined model reached {pct(HI_COMBINED["top1_accuracy"])}% Top-1 and {pct(HI_COMBINED["top5_accuracy"])}% Top-5. On a 500-example Telugu sample, fine-tuned bloom-560m reached 8.6% Top-1 against 22.6% for Hybrid v2.'),
    ('Status', 'All English models and the Hindi and English pretrained comparisons are in progress.'),
    ('Caution', 'Most results use one seed and test sets were reused for reporting, so cross-language differences are indicative only.'),
]
words = sum(len(body.split()) for _, body in ABSTRACT)
assert 100 <= words <= 150, words
slide = new_slide('Abstract',
                  f'Abstract length is {words} words. Stress three things: one method for all languages, measured results on unseen '
                  'data, and clear caveats. Abbreviations: Top-k means the correct word appears among the first k suggestions.',
                  ['Five labelled rows; no chart'])
y = 1.32
for (label, body), height in zip(ABSTRACT, (0.8, 1.15, 1.55, 0.55, 0.8)):
    box(slide, LEFT, y, 1.85, height - 0.1, label, fill=NAVY, color=WHITE, size=16, bold=True, name=f'{label} label')
    text(slide, 2.7, y, 10.03, height - 0.1, body, size=18, anchor=MSO_ANCHOR.MIDDLE, name=f'{label} text')
    y += height + 0.03

# ---------------------------------------------------------------- 3 Introduction

slide = new_slide('Introduction',
                  f'Next-word prediction ranks likely next words from typed context. Google Gboard {c("hard")} is the best-known deployed '
                  'example. Our models output a fixed vocabulary of 20,000 words. The figure is the real project website.',
                  ['Screenshot of the project website (Write page, Telugu)', 'Three-step task strip'])
text(slide, LEFT, 1.4, 6.2, 3.4, [
    '**Task:** rank the most likely next word from the words already typed',
    f'**Uses:** mobile keyboards {c("hard")}, typing assistants, accessibility tools',
    '**Challenge:** Telugu words change endings often, creating many rare forms',
    '**Scope:** Telugu, Hindi and English under one evaluation method',
], size=20, bullets=True, gap=12, name='Introduction points')
steps = [('Typed context', 'Up to 8 words'), ('Model', 'Word probabilities'), ('Suggestions', 'Top-5 ranked words')]
previous = None
for index, (head, sub) in enumerate(steps):
    item = box(slide, LEFT + index * 2.2, 5.05, 1.8, 1.25, [f'**{head}**', sub], fill=PALE_BLUE, size=15, gap=3, name=head)
    if previous:
        arrow(slide, previous, item)
    previous = item
picture = slide.shapes.add_picture(str(HERE / 'assets/website_write.png'), Inches(7.2), Inches(1.5), width=Inches(5.53))
picture.name = 'Website screenshot'
picture.line.color.rgb = rgb(LINE)
describe(picture, 'Project website Write page showing ranked Telugu suggestions from the combined model')
text(slide, 7.2, 5.15, 5.53, 0.4, 'Figure 1. Project website: live Telugu suggestions', size=14, color=MUTED, name='Figure caption')

# ---------------------------------------------------------------- 4 Problem statement

slide = new_slide('Problem Statement',
                  'The problem has two parts: predicting the next word, and measuring that ability fairly. The OOV rate and the seen versus '
                  'unseen gap are measured in this project. OOV means out of vocabulary. Hybrid v2 blends a Bi-LSTM with word counts.',
                  ['Statement panel', 'Three measured challenge cards'])
box(slide, LEFT, 1.4, WIDTH, 1.35,
    'Given the last eight words, rank the next word in Telugu, Hindi and English, and measure performance on text the models never saw.',
    fill=NAVY, color=WHITE, size=22, align=L, inset=(0.3, 0.1, 0.3, 0.1), name='Problem statement')
cards = [
    ('Sparse vocabulary', '7.21%', 'of Telugu test targets fall outside the 20,000-word vocabulary (OOV)'),
    ('Inflation on seen text', '31.97% vs 20.75%', 'Hybrid v2 Top-1 on seen and unseen Telugu data'),
    ('Scores differ widely', '59–99%', 'published figures, often without a stated unseen test set'),
]
for index, (head, stat, body) in enumerate(cards):
    x = LEFT + index * 4.14
    box(slide, x, 3.05, 3.85, 3.6, '', fill=LIGHT, name=f'Card {index + 1}')
    text(slide, x + 0.25, 3.2, 3.35, 0.45, head, size=18, bold=True, color=NAVY, name=f'Card {index + 1} heading')
    text(slide, x + 0.25, 3.85, 3.35, 0.8, stat, size=28, bold=True, color=TEAL, name=f'Card {index + 1} figure')
    text(slide, x + 0.25, 4.8, 3.35, 1.6, body, size=18, name=f'Card {index + 1} text')

# ---------------------------------------------------------------- 5 Motivation

slide = new_slide('Research Motivation',
                  'Four motivations. Bloom-560m is a pretrained 560-million-parameter multilingual language model. The last card is a real '
                  'research question: does a large pretrained model beat a small model trained for this task?',
                  ['2 × 2 grid of motivation cards'])
cards = [
    ('Practical value', 'Faster typing and accessibility support for regional-language users'),
    ('Language coverage', f'Deployed keyboard research is English-centred {c("hard")}; Telugu and Hindi receive less attention'),
    ('Trustworthy evaluation', 'Scores should come from unseen text, with the test method stated in full'),
    ('Model size question', f'Can small task-specific models compete with a 560M-parameter pretrained model {c("bloom")}?'),
]
for index, (head, body) in enumerate(cards):
    x, y = LEFT + (index % 2) * 6.2, 1.45 + (index // 2) * 2.65
    box(slide, x, y, 5.93, 2.4, '', fill=PALE_TEAL if index % 3 == 0 else PALE_BLUE, name=f'Card {index + 1}')
    text(slide, x + 0.3, y + 0.25, 5.33, 0.45, head, size=20, bold=True, color=NAVY, name=f'Card {index + 1} heading')
    text(slide, x + 0.3, y + 0.85, 5.33, 1.4, body, size=20, name=f'Card {index + 1} text')

# ---------------------------------------------------------------- 6 Objectives

slide = new_slide('Research Objectives',
                  'Five measurable objectives. Telugu has met all of them. Hindi has met all except the pretrained comparison. English data '
                  'is prepared and its training is in progress.',
                  ['Five numbered objectives with status tags'])
objectives = [
    ('Prepare', 'grouped 90/5/5 datasets for three languages with zero exact sentence overlap between splits', 'Done'),
    ('Train', 'at least four neural model families per language on one shared pipeline', 'Telugu, Hindi done'),
    ('Measure', 'Top-1, Top-3, Top-5, perplexity and OOV rate on held-out test data', 'Telugu, Hindi done'),
    ('Quantify', 'the gain of hybrid and combined models over the best single model', 'Telugu, Hindi done'),
    ('Compare', 'small task-specific models with a pretrained model on a fixed sample, then deploy on a website', 'Telugu done'),
]
for index, (verb, body, status) in enumerate(objectives):
    y = 1.42 + index * 1.05
    box(slide, LEFT, y + 0.12, 0.62, 0.62, str(index + 1), fill=NAVY, color=WHITE, size=20, bold=True, shape=MSO_SHAPE.OVAL,
        inset=(0, 0, 0, 0), name=f'Number {index + 1}')
    text(slide, 1.45, y, 8.85, 0.9, f'**{verb}** {body}', size=20, anchor=MSO_ANCHOR.MIDDLE, name=f'Objective {index + 1}')
    box(slide, 10.5, y + 0.2, 2.23, 0.48, status, fill=PALE_TEAL, color=TEAL, size=14, bold=True, name=f'Status {index + 1}')

# ---------------------------------------------------------------- 7-8 Literature

widths = [2.25, 2.9, 3.75, 3.23]
header = ['Author & Year', 'Research Method', 'Key Findings', 'Limitations']
slide = new_slide('Literature Review: Keyboards, Telugu and Hindi',
                  'Gboard is the closest benchmark because it reports Top-1 and Top-3 on held-out data. CIFG means coupled input and forget '
                  'gate, an LSTM variant. Two Telugu studies were located; only the first abstract could be read, so no scores are quoted '
                  'for them. Figures for Sharma et al. come from the abstract.',
                  ['Native table: Author & Year, Method, Findings, Limitations (5 papers)'])
table(slide, LEFT, 1.35, widths, [
    header,
    [f'Hochreiter & Schmidhuber, 1997 {c("lstm")}', 'LSTM gated memory cell', 'Retains information across long sequences', 'General sequence model, not word prediction'],
    [f'Hard et al., 2018 {c("hard")}', 'Single-layer CIFG, 670 units, 10,000 words', 'Server model 16.5% Top-1, 27.1% Top-3; n-gram 13.0%, 22.1%', 'English logs only; 10,000-word vocabulary'],
    [f'Sharma et al., 2019 {c("sharma")}', 'LSTM and Bi-LSTM on Hindi', 'Reported 59.46% (LSTM), 81.07% (Bi-LSTM)', 'Abstract only; described elsewhere as validation accuracy'],
    [f'Srinagesh et al., 2021 {c("srinagesh")}', 'RNN with iNLTK for Telugu', 'Next-word prediction for sentence auto-completion', 'Abstract only; dataset and scores not stated there'],
    [f'Vijaya Saraswathi et al., 2022 {c("saraswathi")}', 'RNN mechanism for Telugu', 'Telugu next-word prediction', 'Full text not accessed; scores not verified'],
], row_height=0.9, header_height=0.55, size=15, header_size=15, name='Literature table part 1')

slide = new_slide('Literature Review: Related Languages and Methods',
                  'High scores appear here, but test methods differ, so they cannot be ranked against ours. The Islam et al. paper gives 99% '
                  'in its abstract and 99.74% for 5-grams in its body. The Assamese figure was seen only through citing papers.',
                  ['Native table: Author & Year, Method, Findings, Limitations (5 papers)'])
table(slide, LEFT, 1.35, widths, [
    header,
    [f'Bojanowski et al., 2016 {c("bojanowski")}', 'Word vectors from character n-grams', 'Subword pieces help represent word forms', 'Representation method, not a next-word predictor'],
    [f'Barman & Boruah, 2018 {c("barman")}', 'RNN with LSTM on Assamese', 'Reported 88.20%', 'Seen only through citing papers'],
    [f'Endalie et al., 2022 {c("endalie")}', 'BLSTM-GRU, 63,300 sentences', '78.6% on test set; BLSTM 76.1%', 'Bible text only; Top-k not stated'],
    [f'Islam et al., 2024 {c("islam")}', 'Bi-LSTM on 4- and 5-grams, Bangla news', 'Reported 99% or higher', 'No split or accuracy definition stated'],
    [f'Chhetri et al., 2026 {c("chhetri")}', 'LSTM, Bi-LSTM, GRU; 100,000 sentences', 'GRU best at 74.03%', 'Validation split, Dzongkha only'],
], row_height=0.9, header_height=0.55, size=15, header_size=15, name='Literature table part 2')

# ---------------------------------------------------------------- 9 Research gap

slide = new_slide('Research Gap',
                  'Important correction: the project report says no Telugu paper was found, but later searching located two. So we do not claim '
                  'to be first. The gap is transparent, large-scale evaluation across model families. This reflects our search and is not '
                  'proof that no such study exists.',
                  ['Native table: Earlier work, Gap, This project'])
table(slide, LEFT, 1.4, [2.3, 4.9, 4.93], [
    ['Area', 'Observed in earlier work', 'Response in this project'],
    ['Telugu', f'Earlier RNN studies exist {c("srinagesh", "saraswathi")}; scores not verifiable from accessible text', 'Report Top-1, Top-3, Top-5 on 200,092 unseen examples'],
    ['Test method', f'Scores of 59–99% with splits unclear or unstated {c("sharma", "islam")}', 'Grouped splits, overlap removed, training-only vocabulary'],
    ['Model breadth', f'Mostly recurrent variants {c("chhetri", "endalie")}', 'Recurrent, CNN, Transformer, count-based and combined models'],
    ['Pretrained models', 'No comparison found for this task in the works reviewed', 'bloom-560m tested on a fixed Telugu sample'],
], row_height=0.88, header_height=0.6, size=17, header_size=17, name='Research gap table')
text(slide, LEFT, 6.05, WIDTH, 0.5, 'Based on the project’s literature search; this does not prove that no such study exists.', size=15, color=MUTED, name='Search caveat')

# ---------------------------------------------------------------- 10 Existing system

slide = new_slide('Existing System',
                  'The left column shows the existing approaches and the right column shows their limits. N-gram models count word sequences. '
                  'The bloom limitations are hypotheses from our own test and are discussed later.',
                  ['Two-column comparison with four aligned rows'])
box(slide, LEFT, 1.4, 5.9, 0.6, 'Existing approaches', fill=NAVY, color=WHITE, size=20, bold=True, name='Left heading')
box(slide, 6.83, 1.4, 5.9, 0.6, 'Limitations', fill=TEAL, color=WHITE, size=20, bold=True, name='Right heading')
rows = [
    ('N-gram count models', 'Short context; unseen word sequences need back-off'),
    (f'LSTM keyboards such as Gboard {c("hard")}', 'Reported for English with a 10,000-word vocabulary'),
    (f'Regional-language RNN studies {c("sharma", "endalie", "islam")}', 'Few families compared; test methods vary'),
    (f'Large pretrained language models {c("bloom")}', 'Slow to score: about 5.5 s per example on Apple M5'),
]
for index, (left, right) in enumerate(rows):
    y = 2.15 + index * 1.13
    item = box(slide, LEFT, y, 5.9, 1.0, left, fill=LIGHT, size=19, align=L, inset=(0.25, 0.05, 0.2, 0.05), name=f'Existing {index + 1}')
    other = box(slide, 6.83, y, 5.9, 1.0, right, fill=PALE_TEAL, size=19, align=L, inset=(0.25, 0.05, 0.2, 0.05), name=f'Limitation {index + 1}')
    arrow(slide, item, other)

# ---------------------------------------------------------------- 11 Proposed system

slide = new_slide('Proposed System',
                  'One pipeline handles all three languages, so differences between models are not caused by different data handling. The '
                  'combined model adds model probabilities using fixed weights chosen on validation data. The website has Write, Compare, '
                  'Generate and Results pages.',
                  ['Four feature cards', 'Combined-model formula strip'])
cards = [
    ('One pipeline', 'Same splits, vocabulary size and context length for all languages'),
    ('Several model families', f'LSTM {c("lstm")}, Bi-LSTM {c("bilstm")}, GRU {c("gru")}, gated CNNs, Transformer {c("transformer")}'),
    ('Hybrid and combined models', 'Blend network and word-count probabilities; weights chosen on validation data'),
    ('Working website', 'FastAPI server with Write, Compare, Generate and Results pages'),
]
for index, (head, body) in enumerate(cards):
    x, y = LEFT + (index % 2) * 6.2, 1.42 + (index // 2) * 2.1
    box(slide, x, y, 5.93, 1.9, '', fill=LIGHT, name=f'Card {index + 1}')
    text(slide, x + 0.3, y + 0.18, 5.33, 0.42, head, size=20, bold=True, color=NAVY, name=f'Card {index + 1} heading')
    text(slide, x + 0.3, y + 0.68, 5.33, 1.1, body, size=18, name=f'Card {index + 1} text')
box(slide, LEFT, 5.7, WIDTH, 0.95, 'Principle: choose on validation data, report once on the held-out test split, and count OOV targets as misses',
    fill=NAVY, color=WHITE, size=19, inset=(0.3, 0.05, 0.3, 0.05), name='Design principle')

# ---------------------------------------------------------------- 12 Architecture

slide = new_slide('Proposed System Architecture',
                  'Read left to right for training and bottom row for use. Asterisk: GRU and Transformer were trained for Hindi and English, '
                  'not for Telugu. Telugu added Bi-LSTM fine-tuning instead. All boxes match the code in src and the records in outputs.',
                  ['Editable six-layer architecture diagram built from shapes and connectors'])
columns = [
    ('Data sources', [f'Telugu: Alpaca 28,910 {c("te_alpaca")}, GPTeacher 43,614 {c("te_gpteacher")}', f'Hindi: 51,760 rows {c("hi_alpaca")}', f'English: 51,760 rows {c("en_alpaca")}']),
    ('Preprocessing', ['Clean, split sentences', 'Hash split 90/5/5 by instruction', 'Remove sentence overlap', 'Vocabulary 20,000', '8-word prefix + target']),
    ('Models', ['LSTM, Bi-LSTM, GRU*', 'Gated CNN, word-character CNN', 'Transformer*', 'Word counts']),
    ('Blending', ['Hybrid: network + counts', 'Combined: weighted sum', 'Weights chosen on validation']),
    ('Evaluation', ['Top-1, Top-3, Top-5', 'Perplexity', 'OOV rate', 'Held-out test split']),
]
column_width, gap = 2.13, 0.37
headers = []
for index, (head, items) in enumerate(columns):
    x = LEFT + index * (column_width + gap)
    headers.append(box(slide, x, 1.32, column_width, 0.5, head, fill=NAVY, color=WHITE, size=16, bold=True, name=head))
    box(slide, x, 1.88, column_width, 3.32, [f'{item}' for item in items], fill=LIGHT, size=14, align=L, anchor=MSO_ANCHOR.MIDDLE,
        inset=(0.12, 0.06, 0.08, 0.06), gap=9, name=f'{head} detail')
for first, second in zip(headers, headers[1:]):
    arrow(slide, first, second)
labels = ['User text', 'Tokenize, keep last 8 words', 'Saved models (FastAPI)', 'Ranked Top-5 words', 'Website pages']
previous = None
lane = []
for index, label in enumerate(labels):
    x = LEFT + index * (column_width + gap)
    item = box(slide, x, 5.6, column_width, 0.9, label, fill=PALE_TEAL, size=15, bold=index == 2, name=label)
    if previous:
        arrow(slide, previous, item, color=TEAL)
    previous = item
    lane.append(item)
model_detail = [shape for shape in slide.shapes if shape.name == 'Models detail'][0]
arrow(slide, model_detail, lane[2], start_site=2, end_site=0, color=TEAL)
text(slide, LEFT, 6.54, WIDTH, 0.3, '*Trained for Hindi and English. Telugu used LSTM, Bi-LSTM and CNN models plus fine-tuned Bi-LSTMs.', size=13, color=MUTED, name='Architecture note')

# ---------------------------------------------------------------- 13 Workflow

slide = new_slide('Complete System Workflow',
                  'This slide shows the order of events, not the components. The upper lane runs once, offline. The lower lane runs each time '
                  'the user types. Inference uses saved weights only; typed text does not update the models. PAD and UNK are the padding and '
                  'unknown-word tokens.',
                  ['Editable two-lane flowchart: offline training and online prediction, with a decision branch'])
text(slide, LEFT, 1.3, 7, 0.4, 'Offline: experiment workflow', size=18, bold=True, color=NAVY, name='Offline lane title')
steps = ['1. Verify source checksums', '2. Prepare splits, vocabulary', '3. Train each model', '4. Keep best validation checkpoint', '5. Select blend weights', '6. Score test split']
width, gap = 1.72, 0.362
previous = None
for index, label in enumerate(steps):
    item = box(slide, LEFT + index * (width + gap), 1.75, width, 1.2, label, fill=PALE_BLUE, size=15, name=label)
    if previous:
        arrow(slide, previous, item)
    previous = item
text(slide, LEFT, 3.2, 7, 0.4, 'Online: prediction workflow', size=18, bold=True, color=NAVY, name='Online lane title')
one = box(slide, LEFT, 3.75, 1.72, 1.2, 'User types text', fill=PALE_TEAL, size=15, name='User input')
two = box(slide, 2.682, 3.75, 1.72, 1.2, 'Tokenize, keep last 8 words', fill=PALE_TEAL, size=15, name='Tokenize')
three = box(slide, 4.62, 3.6, 2.1, 1.5, 'Known word?', fill=NAVY, color=WHITE, size=15, bold=True, shape=MSO_SHAPE.DIAMOND, inset=(0, 0, 0, 0), name='Decision')
four = box(slide, 6.95, 3.75, 1.72, 1.2, 'Map to IDs, left-pad', fill=PALE_TEAL, size=15, name='Encode')
five = box(slide, 9.0, 3.75, 1.72, 1.2, 'Selected model', fill=PALE_TEAL, size=15, name='Model')
six = box(slide, 11.01, 3.75, 1.72, 1.2, 'Top-5 without PAD, UNK', fill=PALE_TEAL, size=15, name='Suggestions')
no = box(slide, 4.62, 5.55, 2.1, 0.95, 'No suggestions', fill=PALE_AMBER, color=AMBER, size=15, bold=True, name='No suggestions')
for first, second in ((one, two), (two, three), (three, four), (four, five), (five, six)):
    arrow(slide, first, second, color=TEAL)
arrow(slide, three, no, start_site=2, end_site=0, color=AMBER)
text(slide, 6.62, 3.95, 0.4, 0.3, 'Yes', size=13, color=TEAL, bold=True, name='Yes label')
text(slide, 5.77, 5.12, 0.4, 0.3, 'No', size=13, color=AMBER, bold=True, name='No label')
text(slide, 6.95, 5.55, 5.78, 0.95, 'Saved weights are used for inference. User text is not used for retraining.', size=17, anchor=MSO_ANCHOR.MIDDLE, name='Inference note')

# ---------------------------------------------------------------- 14 Methodology

slide = new_slide('Research Methodology',
                  'Six stages. Top-k counts how often the true word is among the k highest-ranked words. PAD and UNK are removed from ranking, '
                  'so an OOV target can never score. In cross-entropy, OOV targets map to UNK. Perplexity is the exponential of cross-entropy; '
                  'lower is better. Weights were picked on validation Top-1: 110 candidates for Telugu and 287 for Hindi, screened on 60,000 '
                  'validation examples.',
                  ['Six numbered stages', 'Equation panel with editable text: cross-entropy, perplexity, Top-k, OOV rate, combined model'])
stages = [
    ('Prepare', 'grouped splits, overlap removed'),
    ('Encode', 'prefix of up to 8 words, then target'),
    ('Train', 'minimize cross-entropy'),
    ('Select', 'best checkpoint on validation'),
    ('Combine', 'choose weights on validation'),
    ('Evaluate', 'held-out test split'),
]
for index, (verb, body) in enumerate(stages):
    y = 1.42 + index * 0.87
    box(slide, LEFT, y + 0.06, 0.58, 0.58, str(index + 1), fill=NAVY, color=WHITE, size=18, bold=True, shape=MSO_SHAPE.OVAL, inset=(0, 0, 0, 0), name=f'Stage number {index + 1}')
    text(slide, 1.35, y, 4.75, 0.72, f'**{verb}:** {body}', size=18, anchor=MSO_ANCHOR.MIDDLE, name=f'Stage {index + 1}')
box(slide, 6.3, 1.42, 6.43, 5.2, '', fill=LIGHT, name='Equation panel')
equations = [
    ('Cross-entropy', 'CE = −(1/N) Σ_{i} log p(y_{i} | x_{i})'),
    ('Perplexity', 'PP = exp(CE)'),
    ('Top-k accuracy', 'Top-k = (1/N) Σ_{i} 1[ y_{i} ∈ T_{k}(x_{i}) ]'),
    ('OOV rate', 'OOV = (1/N) Σ_{i} 1[ y_{i} = UNK ]'),
    ('Combined model', 'p(w | x) = Σ_{m} λ_{m} p_{m}(w | x),   Σ_{m} λ_{m} = 1'),
]
for index, (label, equation) in enumerate(equations):
    y = 1.55 + index * 0.84
    text(slide, 6.6, y, 5.9, 0.3, label, size=14, color=MUTED, name=f'{label} label')
    text(slide, 6.6, y + 0.3, 5.9, 0.45, equation, size=20, color=NAVY, name=f'{label} equation')
text(slide, 6.6, 5.82, 5.9, 0.7, 'T_{k}(x): k highest-ranked words, excluding PAD and UNK. OOV targets therefore always count as misses.', size=14, name='Equation note')

# ---------------------------------------------------------------- 15 Model architecture

slide = new_slide('Proposed Model Architecture',
                  'Upper part: the Telugu combined model. Weights are 0.3 LSTM, 0.3 Bi-LSTM v2, 0.1 Bi-LSTM v3 and 0.3 counts. Counts back off '
                  'from long to short contexts, with confidence n/(n+5) for three- and four-word contexts. Lower part: the Hindi Transformer, '
                  'the highest Hindi single model. It uses a causal mask and predicts from the last position.',
                  ['Editable layer diagram of the combined model with dimensions and weights', 'Editable Transformer layer diagram'])
ids = box(slide, LEFT, 1.75, 1.45, 2.65, ['**Input**', '8 word IDs', '(B × 8)'], fill=NAVY, color=WHITE, size=15, gap=4, name='Input IDs')
lanes = [
    ('LSTM, λ = 0.3', ['Embedding 64', 'LSTM 128', 'Linear 20,000']),
    ('Bi-LSTM v2 and v3, λ = 0.3 and 0.1', ['Embedding 64', 'Bi-LSTM 128 + 128', 'Linear 20,000']),
    ('Word counts, λ = 0.3', ['1–2 word back-off', '3–4 word smoothing', 'Count probabilities']),
]
blend = box(slide, 10.0, 1.75, 1.35, 2.65, ['**Weighted**', '**sum**'], fill=TEAL, color=WHITE, size=15, gap=2, name='Weighted sum')
top5 = box(slide, 11.62, 1.75, 1.11, 2.65, ['**Top-5**', 'words'], fill=NAVY, color=WHITE, size=15, gap=4, name='Top-5 output')
arrow(slide, blend, top5, color=TEAL)
for row, (label, layers) in enumerate(lanes):
    y = 1.32 + row * 0.98
    text(slide, 2.45, y, 7.3, 0.3, label, size=13, color=MUTED, name=f'{label} label')
    previous = None
    for index, layer in enumerate(layers):
        item = box(slide, 2.45 + index * 2.5, y + 0.3, 2.2, 0.58, layer, fill=PALE_BLUE if row < 2 else PALE_TEAL, size=14, name=f'{label}: {layer}')
        if previous:
            arrow(slide, previous, item)
        previous = item
text(slide, LEFT, 4.5, 11, 0.36, f'Hindi Transformer: highest single-model Top-1, {HI["transformer"]["parameter_count"]:,} parameters', size=16, bold=True, color=NAVY, name='Transformer heading')
layers = [('Input', '8 word IDs'), ('Embedding', 'Word + position, 128'), ('Encoder × 4', '4 heads, 512 feed-forward, causal mask'),
          ('LayerNorm', 'Last position'), ('Linear', '20,000 logits')]
sizes = [1.6, 2.5, 3.5, 1.75, 1.7]
x, previous = LEFT, None
for (head, detail), size in zip(layers, sizes):
    item = box(slide, x, 4.95, size, 1.1, [f'**{head}**', detail], fill=LIGHT, line=LINE, size=14, gap=2, name=f'Transformer {head}')
    if previous:
        arrow(slide, previous, item)
    previous = item
    x += size + 0.27
text(slide, LEFT, 6.2, WIDTH, 0.55, f'Hindi Combined weights: Transformer {HI_WEIGHTS["transformer"]}, Bi-LSTM {HI_WEIGHTS["bilstm"]}, counts {HI_WEIGHTS["counts"]}, word-character CNN {HI_WEIGHTS["wordchar_cnn"]}. Parameters: LSTM {TE["lstm"]["parameter_count"]:,}; Bi-LSTM {TE["bilstm"]["parameter_count"]:,}.', size=14, color=MUTED, name='Parameter note')

# ---------------------------------------------------------------- 16 Dataset

splits = EN_PROVENANCE['splits']
slide = new_slide('Dataset Description',
                  'All counts come from the provenance files. Hindi has fewer rows than Telugu but more examples, because its sentences are '
                  'longer: 5.2 million versus 3.48 million. Hindi is machine-translated with NLLB-1.3B, so it is not naturally written text. '
                  'Project files do not record how the Telugu sources were produced. About 3.83% of Hindi test examples sit in translation '
                  'repetition loops.',
                  ['Native table: parameter and value per language'])
table(slide, LEFT, 1.33, [3.13, 3.1, 3.0, 2.9], [
    ['Parameter', 'Telugu', 'Hindi', 'English'],
    ['Source', f'Alpaca, GPTeacher {c("te_alpaca", "te_gpteacher")}', f'alpaca-cleaned-hindi {c("hi_alpaca")}', f'alpaca-cleaned {c("en_alpaca")}'],
    ['Text type', 'Instruction-style', f'Machine-translated {c("nllb")}', 'Original English'],
    ['Source records', '72,524', '51,760', '51,760'],
    ['Training examples', '3,483,073', '5,203,483', f'{splits["train"]["sequences"]:,}'],
    ['Validation examples', '198,928', '293,336', f'{splits["validation"]["sequences"]:,}'],
    ['Test examples', '200,092', '296,193', f'{splits["test"]["sequences"]:,}'],
    ['Test sentences', '18,662', '16,815', f'{splits["test"]["sentences"]:,}'],
    ['Vocabulary / context', '20,000 / 8 words', '20,000 / 8 words', '20,000 / 8 words'],
    ['Test OOV rate', pct(TE['lstm']['target_oov_rate']) + '%', pct(HI['lstm']['target_oov_rate']) + '%', IN_PROGRESS],
], row_height=0.49, size=16, header_size=16, name='Dataset table')
text(slide, LEFT, 6.45, WIDTH, 0.4, 'Hindi has fewer rows than Telugu but more examples (5.2M vs 3.48M), because its sentences are longer.', size=15, color=MUTED, name='Dataset note')

# ---------------------------------------------------------------- 17 Setup

slide = new_slide('Experimental Setup',
                  'Only recorded settings are shown. Telugu baselines ran on CPU with 4 threads; the machine itself is not recorded. Hindi ran '
                  'on the Apple M5 laptop. Recurrent models used 3 Hindi epochs and CNN and Transformer used 5, so Hindi epochs are unequal. '
                  f'Adam is from {c("adam")}. The Transformer pilot used 1 million training examples for one epoch and compared 2 and 4 layers '
                  'at two learning rates on validation data.',
                  ['Native table: hardware, software and training settings'])
table(slide, LEFT, 1.33, [3.0, 9.13], [
    ['Item', 'Recorded configuration'],
    ['Hardware: Telugu', 'CPU, 4 threads (machine model not recorded)'],
    ['Hardware: Hindi, English', 'Apple M5, 10 cores, 16 GB; CPU for recurrent, Apple GPU for CNN and Transformer'],
    ['Software', f'Python, PyTorch, FastAPI, HTML/CSS/JavaScript; transformers and peft for LoRA {c("lora")}'],
    ['Optimizer', f'Adam {c("adam")}, learning rate 0.001, batch 512, seed 42; Transformer: AdamW'],
    ['Epochs', 'Telugu baselines 5; Hindi recurrent 3; Hindi CNN and Transformer 5'],
    ['Fine-tuning (Telugu)', 'Bi-LSTM v2: rate 0.0003 to epoch 10; v3: AdamW, 0.0001, dropout 0.15 to epoch 14'],
    ['Transformer choice', 'Pilot on validation data selected 4 layers, learning rate 0.001'],
    ['Pretrained test', 'bloom-560m, LoRA r = 16, 2,500 steps, 20,000 sentences, learning rate 5e-5'],
], row_height=0.58, size=16, header_size=16, name='Experimental setup table')

# ---------------------------------------------------------------- 18 Telugu results

v2_te = TE['v2']
rows = [['Model', 'Top-1', 'Top-3', 'Top-5', 'PP']]
for label, metrics in (('LSTM', TE['lstm']), ('Bi-LSTM', TE['bilstm']), ('Word CNN', TE['cnn']), ('Fine-tuned word CNN', TE_CNN2),
                       ('Word-character CNN', TE['wordchar']), ('Bi-LSTM v2', TE['v2']), ('Bi-LSTM v3', TE['v3'])):
    rows.append([label, pct(metrics['top1_accuracy']), pct(metrics['top3_accuracy']), pct(metrics['top5_accuracy']), f'{metrics["vocabulary_perplexity"]:.2f}'])
rows.append(['Hybrid v1', pct(TE_H1['top1_accuracy']), '–', pct(TE_H1['top5_accuracy']), '–'])
rows.append(['Hybrid v2', pct(TE_ENS['hybrid_v2']['top1_accuracy']), pct(TE_ENS['hybrid_v2']['top3_accuracy']), pct(TE_ENS['hybrid_v2']['top5_accuracy']), '–'])
rows.append(['Combined', pct(TE_ENS['ensemble']['top1_accuracy']), pct(TE_ENS['ensemble']['top3_accuracy']), pct(TE_ENS['ensemble']['top5_accuracy']), '–'])
slide = new_slide('Experimental Results: Telugu',
                  'All rows use the same 200,092 test examples. Accuracy is in percent; PP is perplexity, lower is better. A dash means not '
                  'measured. Combined is highest on all three accuracy measures. The word-character CNN has the lowest measured perplexity. '
                  'Five seeds: LSTM Top-1 17.94 ± 0.07 and Bi-LSTM 18.72 ± 0.10.',
                  ['Native table: ten Telugu models', 'Native line chart: validation Top-1 per epoch from history files'])
table(slide, LEFT, 1.3, [2.75, 1.0, 1.0, 1.0, 1.05], rows, row_height=0.4, size=15, header_size=15, aligns=[L, R, R, R, R], highlight=(10,), name='Telugu results table')
line_chart(slide, 7.65, 1.25, 5.08, 4.5, ['1', '2', '3', '4', '5'], [
    ('LSTM', history('telugu_comparison/lstm_history.json')),
    ('Bi-LSTM', history('telugu_comparison/bilstm_history.json')),
    ('Word CNN', history('telugu_cnn/cnn_history.json')),
    ('Word-char CNN', history('telugu_wordchar/wordchar_cnn_history.json')),
], [NAVY, TEAL, BLUE, AMBER], 12, 20, 2, 'Telugu validation Top-1 per epoch for four baseline models')
text(slide, LEFT, 5.8, 6.8, 0.4, 'Accuracy in %. PP: perplexity, lower is better. –: not measured.', size=14, color=MUTED, name='Table key')
text(slide, 7.65, 5.78, 5.08, 0.9, f'Test OOV {pct(TE["lstm"]["target_oov_rate"])}% for all rows. Five seeds: LSTM {SEEDS["LSTM"]["Top-1"]["mean"]:.2f} ± {SEEDS["LSTM"]["Top-1"]["std"]:.2f}, Bi-LSTM {SEEDS["Bi-LSTM"]["Top-1"]["mean"]:.2f} ± {SEEDS["Bi-LSTM"]["Top-1"]["std"]:.2f}.', size=14, color=MUTED, name='Telugu note')

# ---------------------------------------------------------------- 19 Hindi results

rows = [['Model', 'Top-1', 'Top-3', 'Top-5', 'PP', 'OOV']]
for label, metrics in (('Combined', HI_COMBINED), ('Hybrid', HI_HYBRID), ('Transformer', HI['transformer']), ('Bi-LSTM', HI['bilstm']),
                       ('Word-character CNN', HI['wordchar']), ('Gated CNN', HI['cnn']), ('LSTM', HI['lstm']), ('GRU', HI['gru'])):
    rows.append([label, pct(metrics['top1_accuracy']), pct(metrics['top3_accuracy']), pct(metrics['top5_accuracy']), f'{metrics["vocabulary_perplexity"]:.2f}', pct(metrics['target_oov_rate'])])
slide = new_slide('Experimental Results: Hindi',
                  'All rows use the same 296,193 test examples, one seed, machine-translated text. Combined is highest on every measure, '
                  f'with {pct(HI_COMBINED["validation_top1"])}% validation Top-1, and also has the lowest perplexity. Hybrid leads Transformer on Top-1 and Top-3, but not on Top-5. '
                  'Recurrent models stopped at epoch 3 while still improving, so they had fewer epochs than CNN and Transformer.',
                  ['Native table: eight Hindi models', 'Native line chart: validation Top-1 per epoch from history files'])
table(slide, LEFT, 1.3, [2.55, 1.0, 1.0, 1.0, 1.0, 0.85], rows, row_height=0.43, size=15, header_size=15, aligns=[L, R, R, R, R, R], highlight=(1,), name='Hindi results table')


def padded(path):
    values = history(path)
    return values + [None] * (5 - len(values))


line_chart(slide, 8.45, 1.25, 4.28, 4.5, ['1', '2', '3', '4', '5'], [
    ('Transformer', padded('hindi_families/transformer_history.json')),
    ('Bi-LSTM', padded('hindi_comparison/bilstm_history.json')),
    ('Word-char CNN', padded('hindi_families/wordchar_cnn_history.json')),
    ('LSTM', padded('hindi_comparison/lstm_history.json')),
], [TEAL, NAVY, AMBER, BLUE], 26, 34, 2, 'Hindi validation Top-1 per epoch for four models')
text(slide, LEFT, 5.3, 7.4, 1.3, [
    'Accuracy and OOV in %. PP: perplexity, lower is better.',
    f'Without translation repetition loops (3.83% of examples): Combined Top-1 {pct(HI_COMBINED["without_loops"]["top1_accuracy"])}%.',
    'Single seed, machine-translated text, test set reused for reporting.'], size=14, color=MUTED, gap=3, name='Hindi note')
text(slide, 8.45, 5.78, 4.28, 0.6, 'Recurrent models ran 3 epochs; others ran 5.', size=14, color=MUTED, name='Epoch note')

# ---------------------------------------------------------------- 20 Comparative analysis

slide = new_slide('Comparative Performance Analysis',
                  'The chart shows only models trained in both languages, using the baseline runs. It is not a language ranking: data sources '
                  'differ, Hindi is machine-translated, OOV differs (7.21% vs 1.04%) and Telugu recurrent models had 5 epochs against 3. Gboard '
                  'is English logs with a 10,000-word vocabulary, so it is a guide only. The 500-example sample is slightly easier than the full '
                  'Telugu test set.',
                  ['Native clustered bar chart: Top-1 by model and language', 'Native table: best results with evaluation conditions'])
data = CategoryChartData()
data.categories = ['LSTM', 'Bi-LSTM', 'Gated CNN', 'Word-char']
data.add_series('Telugu', [round(TE[k]['top1_accuracy'] * 100, 2) for k in ('lstm', 'bilstm', 'cnn', 'wordchar')])
data.add_series('Hindi', [round(HI[k]['top1_accuracy'] * 100, 2) for k in ('lstm', 'bilstm', 'cnn', 'wordchar')])
frame = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(LEFT), Inches(1.25), Inches(5.3), Inches(4.45), data)
frame.name = 'Chart: Top-1 by model and language'
describe(frame, 'Top-1 test accuracy for four shared models in Telugu and Hindi; English in progress')
chart = frame.chart
style_chart(chart, 13)
chart.value_axis.minimum_scale, chart.value_axis.maximum_scale, chart.value_axis.major_unit = 0, 40, 10
chart.value_axis.tick_labels.number_format, chart.value_axis.tick_labels.number_format_is_linked = '0', False
axis_title(chart.value_axis, 'Test Top-1 (%)')
skip = etree.SubElement(chart.category_axis._element, qn('c:tickLblSkip'))
skip.set('val', '1')
chart.category_axis._element.find(qn('c:noMultiLvlLbl')).addprevious(skip)
plot = chart.plots[0]
plot.gap_width, plot.overlap = 60, -5
for plot_series, color in zip(plot.series, (NAVY, TEAL)):
    plot_series.format.fill.solid()
    plot_series.format.fill.fore_color.rgb = rgb(color)
plot.has_data_labels = True
plot.data_labels.number_format, plot.data_labels.number_format_is_linked = '0.0', False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.font.size = Pt(12)
plot.data_labels.show_value = True
table(slide, 6.15, 1.33, [2.45, 1.05, 1.2, 0.95, 0.93], [
    ['System', 'Test size', 'Top-1', 'Top-3', 'Top-5'],
    ['Telugu Combined', '200,092', pct(TE_ENS['ensemble']['top1_accuracy']), pct(TE_ENS['ensemble']['top3_accuracy']), pct(TE_ENS['ensemble']['top5_accuracy'])],
    ['Hindi Combined', '296,193', pct(HI_COMBINED['top1_accuracy']), pct(HI_COMBINED['top3_accuracy']), pct(HI_COMBINED['top5_accuracy'])],
    ['Hindi Transformer', '296,193', pct(HI['transformer']['top1_accuracy']), pct(HI['transformer']['top3_accuracy']), pct(HI['transformer']['top5_accuracy'])],
    ['English, all models', f'{splits["test"]["sequences"]:,}', IN_PROGRESS, '–', '–'],
    [f'Gboard CIFG {c("hard")}', 'Logs', '16.5', '27.1', '–'],
    ['Telugu Hybrid v2', '500', '22.6', '–', '38.4'],
    ['bloom-560m, LoRA', '500', '8.6', '–', '18.6'],
], row_height=0.53, size=14, header_size=14, aligns=[L, R, R, R, R], name='Comparison table')
text(slide, LEFT, 5.78, WIDTH, 0.95, [
    '**Conditions:** same method within each language; one seed for Hindi; test sets reused for reporting.',
    '**Not a language ranking:** data differ, Hindi is machine-translated, OOV is 7.21% vs 1.04%, and Hindi recurrent models ran fewer epochs.',
], size=14, color=MUTED, gap=3, name='Evaluation conditions')

# ---------------------------------------------------------------- 21 Discussion

slide = new_slide('Discussion and Limitations',
                  'The left column is measured. The middle column gives possible explanations that have not been tested, so please read them as '
                  'hypotheses. The right column gives the limits on any claim. Hindi Combined weights were screened on 60,000 validation '
                  'examples and selected on validation data only.',
                  ['Three-column layout separating measured findings, hypotheses and limitations'])
columns = [
    ('Measured findings', NAVY, LIGHT, [
        f'Telugu: Combined beat Hybrid v2 by {(TE_ENS["ensemble"]["top1_accuracy"] - TE_ENS["hybrid_v2"]["top1_accuracy"]) * 100:.2f} Top-1 points',
        f'Hindi: Combined beat Transformer by {(HI_COMBINED["top1_accuracy"] - HI["transformer"]["top1_accuracy"]) * 100:.2f} points',
        'Hindi: word-character CNN got zero blend weight',
        'Telugu: Bi-LSTM beat LSTM in all five seeds',
        'bloom-560m: 8.6% vs Hybrid v2 22.6% (500 examples)',
    ]),
    ('Hypotheses, not tested', TEAL, PALE_TEAL, [
        'Small Telugu gain: models share errors',
        'Larger Hindi gain: Transformer adds diversity',
        'Hindi vs Telugu gap: OOV and data differences',
        'bloom-560m: word splitting by its tokenizer, short fine-tune',
        'More epochs may lift Hindi recurrent models',
    ]),
    ('Limitations', AMBER, PALE_AMBER, [
        'Mostly single seed; test sets reused',
        'Hindi is machine-translated',
        'Fixed 20,000-word vocabulary',
        'One recorded target; other words may be valid',
        'Instruction-style text, not everyday typing',
    ]),
]
for index, (head, color, fill, items) in enumerate(columns):
    x = LEFT + index * 4.14
    box(slide, x, 1.4, 3.85, 0.6, head, fill=color, color=WHITE, size=18, bold=True, name=head)
    box(slide, x, 2.05, 3.85, 4.65, items, fill=fill, size=18, align=L, anchor=MSO_ANCHOR.TOP, inset=(0.2, 0.18, 0.15, 0.1), gap=11, bullets=True, name=f'{head} list')

# ---------------------------------------------------------------- 22 Conclusion

slide = new_slide('Conclusion and Future Scope',
                  'Conclusions are limited to what was measured. The first future step finishes the running work. We do not claim novelty or '
                  'superiority over published work, because languages, data and test methods differ.',
                  ['Two-column conclusions and future work'])
box(slide, LEFT, 1.4, 5.93, 0.6, 'Conclusion', fill=NAVY, color=WHITE, size=20, bold=True, name='Conclusion heading')
box(slide, 6.8, 1.4, 5.93, 0.6, 'Future scope', fill=TEAL, color=WHITE, size=20, bold=True, name='Future heading')
box(slide, LEFT, 2.05, 5.93, 4.65, [
    f'End-to-end pipeline and website delivered',
    f'Telugu Combined: {pct(TE_ENS["ensemble"]["top1_accuracy"])}% Top-1, {pct(TE_ENS["ensemble"]["top5_accuracy"])}% Top-5',
    f'Hindi Combined: {pct(HI_COMBINED["top1_accuracy"])}% Top-1, {pct(HI_COMBINED["top5_accuracy"])}% Top-5',
    'Small task-specific models beat fine-tuned bloom-560m on a 500-example Telugu sample',
    'Scores depend strongly on test method',
], fill=LIGHT, size=19, align=L, anchor=MSO_ANCHOR.TOP, inset=(0.25, 0.2, 0.2, 0.1), gap=12, bullets=True, name='Conclusion list')
box(slide, 6.8, 2.05, 5.93, 4.65, [
    'Complete all English models',
    'Run pretrained models on Hindi and English',
    'Repeat seeds and use a fresh test set',
    'Train an ungated word-character CNN',
    'Add varied Telugu text; measure keystrokes saved',
], fill=PALE_TEAL, size=19, align=L, anchor=MSO_ANCHOR.TOP, inset=(0.25, 0.2, 0.2, 0.1), gap=12, bullets=True, name='Future list')

# ---------------------------------------------------------------- 23-24 References

for key in ('adam',):
    c(key)
missing = set(REFERENCES) - set(cited)
assert not missing, missing
split = 11
for part, (start, end) in enumerate(((0, split), (split, len(cited))), 1):
    slide = new_slide(f'References ({part} of 2)', 'References follow IEEE style and are numbered by first citation in this deck.',
                      ['Numbered IEEE reference list'])
    lines = [f'[{index + 1}]  {REFERENCES[cited[index]]}' for index in range(start, end)]
    text(slide, LEFT, 1.3, WIDTH, 5.5, lines, size=14, gap=5, name='Reference list')

prs.core_properties.title = 'Multilingual Next-Word Prediction'
prs.core_properties.subject = 'UROP final evaluation'
prs.core_properties.author = ''
prs.core_properties.last_modified_by = ''
prs.core_properties.comments = ''
prs.core_properties.created = prs.core_properties.modified = datetime.now()

# python-pptx leaves the notes master out of presentation.xml; Keynote then refuses to open the file.
notes_master_rid = prs.part.relate_to(prs.part.notes_master_part, RT.NOTES_MASTER)
notes_master_list = etree.Element(qn('p:notesMasterIdLst'))
etree.SubElement(notes_master_list, qn('p:notesMasterId')).set(qn('r:id'), notes_master_rid)
prs.part._element.find(qn('p:sldMasterIdLst')).addnext(notes_master_list)
prs.save(HERE / 'Research_Presentation.pptx')

# ---------------------------------------------------------------- Markdown deliverables

QUESTIONS = [
    ('Why is your accuracy so far below the 59–99% in other papers?',
     'Those papers use different languages and data, and several state no separate unseen test set. Ours counts OOV targets as wrong and uses grouped unseen text. On seen data Hybrid v2 scores 31.97% Top-1, against 20.75% unseen.'),
    ('Your report says no Telugu paper exists. Is that true?',
     f'No. The report line is out of date. Two earlier Telugu studies were located {c("srinagesh", "saraswathi")}. We could read only one abstract, so we quote no scores from them. We do not claim to be first.'),
    ('Is Hindi easier than Telugu, since it scores higher?',
     'We cannot say. The data source differs, Hindi is machine-translated, its OOV rate is 1.04% against 7.21%, and it has more examples. The comparison is indicative only.'),
    ('Why does a Bi-LSTM not cheat by seeing the future?',
     'It reads only the observed prefix in both directions. The target word is never in its input.'),
    ('How do you know there is no data leakage?',
     'Splits are made by a hash of the normalized instruction, so all text from one instruction stays in one split, and exact repeated sentences are removed. Near-duplicates with different wording are not ruled out.'),
    ('What does perplexity tell you that accuracy does not?',
     'It measures how much probability the model gives to the true word across the whole distribution. It is the exponential of cross-entropy, and lower is better. OOV targets map to UNK in this measure.'),
    ('Is a 0.31 point gain from the combined model meaningful?',
     'It is small, and the combined model has not been repeated across seeds. Seed spread for LSTM and Bi-LSTM is about 0.1 points, which suggests a real but modest gain. The Hindi gain over the best single model is larger, at 1.75 points.'),
    ('Why did the large pretrained model do worse?',
     'Measured: 8.6% against 22.6% on a 500-example sample. Likely reasons, not tested: its tokenizer splits Telugu words into pieces and the fine-tune covered under 1% of the examples. Its word ranking is also approximate.'),
    ('Why were the Hindi models trained for different numbers of epochs?',
     'Recurrent models used 3 epochs to fit the available time, and were still improving. CNN and Transformer used 5. So Hindi family differences are not on equal training budgets.'),
    ('Why is English not finished?',
     'Its data is prepared and training is running. No English score is shown because none exists yet.'),
    ('Why did the word-character CNN get zero weight in Hindi Combined?',
     'The validation search gave it no weight, so it added nothing beyond the Transformer, Bi-LSTM and counts on that measure. This does not show that it has no value in other settings.'),
    ('Is the Gboard comparison fair?',
     'Only as a guide. It uses the same type of measure, but real English typing with a 10,000-word vocabulary.'),
]
notes = ['# Presentation Speaker Notes', '', 'Short notes per slide, followed by likely professor questions.', '']
for item in SLIDES:
    notes += [f'## Slide {item["number"]}: {item["title"]}', '', item['notes'], '']
notes += ['## Likely Professor Questions', '']
for question, answer in QUESTIONS:
    notes += [f'**Q: {question}**', '', f'A: {answer}', '']
(HERE / 'Presentation_Speaker_Notes.md').write_text('\n'.join(notes), encoding='utf-8')

outline = ['# Research Presentation Outline', '', f'{len(SLIDES)} slides, 16:9. Diagrams, tables and charts are native and editable.', '',
           'Changes from the planned 21 slides: Literature Review, Experimental Results and References each take two slides.', '']
for item in SLIDES:
    outline += [f'## {item["number"]}. {item["title"]}', ''] + [f'- {visual}' for visual in item['visuals']] + ['']
(HERE / 'Research_Presentation_Outline.md').write_text('\n'.join(outline), encoding='utf-8')
print(len(SLIDES), 'slides; abstract words:', words)
