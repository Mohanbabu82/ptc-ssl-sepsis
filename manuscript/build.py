import re, json, os, shutil, subprocess, zipfile, html
from PIL import Image
import content as C

TPL = 'tpl'
OUT = 'out'
if os.path.exists(OUT): shutil.rmtree(OUT)
shutil.copytree(TPL, OUT)

doc = open(f'{TPL}/word/document.xml', encoding='utf8').read()
head = doc[:doc.find('<w:body>') + len('<w:body>')]
body = doc[doc.find('<w:body>') + len('<w:body>'):]
items = re.findall(r'<w:p[ >].*?</w:p>|<w:tbl>.*?</w:tbl>', body, flags=re.S)
final_sect = re.search(r'<w:sectPr(?:(?!<w:sectPr).)*?</w:sectPr></w:body>', doc, flags=re.S).group(0).replace('</w:body>', '')

def esc(t): return html.escape(t, quote=False)

# ---------------- citations ----------------
order = []
def cite_num(k):
    if k not in C.REFS: raise KeyError(k)
    if k not in order: order.append(k)
    return order.index(k) + 1

# ---------------- inline runs ----------------
TOK = re.compile(r'(\{\{.+?\}\}|\*\*.+?\*\*|\*[^*\s][^*]*?\*|_\{[^}]*\}|\^\{[^}]*\}|\[@\w+\])')
def runs(text, rpr_base='', basic=False, hl=False, b=False, i=False):
    out = []
    pos = 0
    pat = re.compile(r'(_\{[^}]*\}|\^\{[^}]*\})') if basic else TOK
    for m in pat.finditer(text):
        if m.start() > pos: out.append(run(text[pos:m.start()], rpr_base, b, i, hl))
        t = m.group(0)
        if t.startswith('{{'):
            out += runs(t[2:-2], rpr_base, basic, True, b, i)
        elif t.startswith('**'):
            out += runs(t[2:-2], rpr_base, basic, hl, True, i)
        elif t.startswith('*'):
            out += runs(t[1:-1], rpr_base, basic, hl, b, not i)
        elif t.startswith('_{'):
            out.append(run(t[2:-1], rpr_base, b, i, hl, 'subscript'))
        elif t.startswith('^{'):
            out.append(run(t[2:-1], rpr_base, b, i, hl, 'superscript'))
        elif t.startswith('[@'):
            out.append(run(f'[{cite_num(t[2:-1])}]', rpr_base, b, False, hl))
        pos = m.end()
    if pos < len(text): out.append(run(text[pos:], rpr_base, b, i, hl))
    return out

def run(t, rpr_base='', b=False, i=False, hl=False, va=None):
    rp = rpr_base + ('<w:b/>' if b else '') + ('<w:i/>' if i else '') + ('<w:highlight w:val="yellow"/>' if hl else '') + (f'<w:vertAlign w:val="{va}"/>' if va else '')
    inner = '<w:br/>'.join(f'<w:t xml:space="preserve">{esc(x)}</w:t>' for x in t.split('\n'))
    return f'<w:r>{"<w:rPr>"+rp+"</w:rPr>" if rp else ""}{inner}</w:r>'

def para(style, text, ppr_extra='', rpr='', basic=False):
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{ppr_extra}</w:pPr>{"".join(runs(text, rpr, basic))}</w:p>'

NONUM = '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>'

# ---------------- section breaks ----------------
PG = '<w:pgSz w:w="11520" w:h="15660" w:code="1"/><w:pgMar w:top="1300" w:right="740" w:bottom="1040" w:left="740" w:header="360" w:footer="640" w:gutter="0"/>'
def sect(cols):
    fr = '<w:footerReference w:type="default" r:id="rId10"/>'
    c = '<w:cols w:num="2" w:space="400"/>' if cols == 2 else '<w:cols w:space="720"/>'
    return f'<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="14" w:lineRule="exact"/><w:sectPr>{fr}<w:type w:val="continuous"/>{PG}{c}<w:docGrid w:linePitch="360"/></w:sectPr></w:pPr></w:p>'

# ---------------- equations via pandoc ----------------
eqs = re.findall(r'^EQ (.+)$', C.BODY, flags=re.M)
md = '\n\n'.join(f'$${e}$$' for e in eqs)
open('eq.md', 'w').write(md)
subprocess.run(['pandoc', 'eq.md', '-o', 'eq.docx'], check=True)
ex = zipfile.ZipFile('eq.docx').read('word/document.xml').decode('utf8')
ex = ex.replace('<m:sty m:val="p" /><m:scr m:val="script" />', '<m:scr m:val="script" /><m:sty m:val="p" />')
ex = re.sub(r'(<m:begChr [^>]*/>)(<m:endChr [^>]*/>)(<m:sepChr [^>]*/>)', r'\1\3\2', ex)
omaths = re.findall(r'<m:oMath>.*?</m:oMath>', ex, flags=re.S)
assert len(omaths) == len(eqs), (len(omaths), len(eqs))

# ---------------- images ----------------
rels = open(f'{OUT}/word/_rels/document.xml.rels', encoding='utf8').read()
img_n = [0]
COLW, WIDEW = 4820, 10040
def image_par(path, wide):
    img_n[0] += 1
    k = img_n[0]
    name = f'ptcssl_fig{k}.png'
    shutil.copy(path, f'{OUT}/word/media/{name}')
    rid = f'rIdFig{k}'
    global rels
    rels = rels.replace('</Relationships>', f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/></Relationships>')
    w, h = Image.open(path).size
    cx = (WIDEW if wide else COLW) * 635
    if not wide: cx = int(cx * 0.98)
    cy = int(cx * h / w)
    return (f'<w:p><w:pPr><w:pStyle w:val="FigureCaption"/><w:keepNext/><w:spacing w:before="120"/><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{1000+k}" name="Figure {k}" descr="Figure {k}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{k}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')

def eq_image(path):
    img_n[0] += 1
    k = img_n[0]
    name = f'ptcssl_eq{k}.png'
    shutil.copy(path, f'{OUT}/word/media/{name}')
    rid = f'rIdFig{k}'
    global rels
    rels = rels.replace('</Relationships>', f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/></Relationships>')
    w, h = Image.open(path).size
    cx, cy = int(w / 600 * 914400), int(h / 600 * 914400)
    return (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{1000+k}" name="Equation {k}"/><wp:cNvGraphicFramePr/><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{k}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')

# ---------------- tables ----------------
TABS = json.load(open('tables.json'))
ROMAN = 'I II III IV V VI VII VIII IX X XI XII XIII XIV'.split()
def fix_cell(s, ti, header):
    if header and ti in C.HEADER_FIX:
        s = C.HEADER_FIX[ti].get(s, s)
    for a, b in C.CELL_FIX: s = s.replace(a, b)
    return s.replace('\\_', '_')

def table_xml(ti):
    title, note, width, rel = C.TABLES[ti]
    rows = TABS[ti]
    tw = WIDEW if width == 'wide' else COLW
    tot = sum(rel); cw = [int(tw * r / tot) for r in rel]; cw[-1] = tw - sum(cw[:-1])
    out = [f'<w:p><w:pPr><w:pStyle w:val="TableTitle"/><w:keepNext/><w:spacing w:before="160" w:after="0"/></w:pPr>{run("TABLE " + ROMAN[ti])}</w:p>',
           f'<w:p><w:pPr><w:pStyle w:val="TableTitle"/><w:keepNext/><w:spacing w:after="60"/></w:pPr>{"".join(runs(title, basic=True))}</w:p>']
    t = [f'<w:tbl><w:tblPr><w:tblW w:w="{tw}" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
         '<w:tblCellMar><w:left w:w="50" w:type="dxa"/><w:right w:w="50" w:type="dxa"/></w:tblCellMar>'
         '<w:tblLook w:val="0000"/></w:tblPr><w:tblGrid>' + ''.join(f'<w:gridCol w:w="{c}"/>' for c in cw) + '</w:tblGrid>']
    for ri, row in enumerate(rows):
        hdr = ri == 0; last = ri == len(rows) - 1
        t.append('<w:tr>' + ('<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>' if hdr else '<w:trPr><w:cantSplit/></w:trPr>'))
        for ci, cell in enumerate(row):
            top = '<w:top w:val="double" w:sz="6" w:space="0" w:color="auto"/>' if hdr else '<w:top w:val="nil"/>'
            bot = '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="auto"/>' if (hdr or last) else '<w:bottom w:val="nil"/>'
            jc = 'left' if ci == 0 or (ti in (1, 9, 10) and ci <= 2 and not hdr and ti != 9) else 'center'
            if ti == 9 and ci == 1: jc = 'left'
            txt = fix_cell(cell, ti, hdr)
            if ti == 2 and not hdr: txt = txt.replace(' [', '\n[')
            if ti == 4 and not hdr and ci < 3: txt = txt.replace('✓', 'Yes').replace('−', 'No')
            rp = '<w:sz w:val="14"/><w:szCs w:val="14"/>' + ('<w:b/>' if hdr else '')
            t.append(f'<w:tc><w:tcPr><w:tcW w:w="{cw[ci]}" w:type="dxa"/><w:tcBorders>{top}<w:left w:val="nil"/>{bot}<w:right w:val="nil"/></w:tcBorders><w:vAlign w:val="center"/></w:tcPr>'
                     f'<w:p><w:pPr><w:spacing w:before="10" w:after="10" w:line="200" w:lineRule="exact"/><w:jc w:val="{jc}"/></w:pPr>{"".join(runs(txt, rp, basic=True))}</w:p></w:tc>')
        t.append('</w:tr>')
    t.append('</w:tbl>')
    out.append(''.join(t))
    if note:
        out.append(f'<w:p><w:pPr><w:pStyle w:val="FootnoteText"/><w:spacing w:before="40" w:after="120"/><w:ind w:firstLine="0"/></w:pPr>{"".join(runs(note, basic=True))}</w:p>')
    else:
        out.append('<w:p><w:pPr><w:pStyle w:val="FootnoteText"/><w:spacing w:before="0" w:after="80"/></w:pPr></w:p>')
    return out, width == 'wide'

# ---------------- body ----------------
out = []
prev = 'heading'
first_h1 = True
eq_i = 0
fig_n = 0
for line in C.BODY.strip('\n').split('\n'):
    if not line.strip():
        continue
    tag, _, rest = line.partition(' ')
    if tag == '#H1':
        style = 'H1ListNoSpace' if first_h1 else 'H1ListSpace'
        first_h1 = False
        out.append(para(style, rest, '<w:keepNext/>')); prev = 'heading'
    elif tag == '#U':
        out.append(para('H1', rest, '<w:keepNext/>')); prev = 'heading'
    elif tag == '#H2':
        out.append(para('H2Cont', rest, '<w:keepNext/>' + NONUM + '<w:ind w:left="0" w:firstLine="0"/>')); prev = 'heading'
    elif tag == 'P':
        out.append(para('PARA' if prev == 'heading' else 'PARAIndent', rest)); prev = 'p'
    elif tag == 'PI':
        out.append(para('PARAIndent', rest)); prev = 'p'
    elif tag == 'EQ':
        eq_i += 1
        mathxml = omaths[eq_i-1]
        if os.environ.get('PREVIEW'):
            mathxml = eq_image(f'fig/eq{eq_i}.png')
        out.append(f'<w:p><w:pPr><w:pStyle w:val="Equation"/><w:tabs><w:tab w:val="clear" w:pos="4584"/><w:tab w:val="center" w:pos="2300"/><w:tab w:val="right" w:pos="4820"/></w:tabs>'
                   f'<w:ind w:left="0"/></w:pPr><w:r><w:tab/></w:r>{mathxml}<w:r><w:rPr><w:i w:val="0"/></w:rPr><w:tab/><w:t>({eq_i})</w:t></w:r></w:p>')
        prev = 'eq'
    elif tag == 'FIG':
        path, size, cap = rest.split('|', 2)
        wide = size == 'wide'
        blk = [image_par(path, wide), para('FigCaption', cap, '<w:spacing w:after="160"/>', rpr='<w:b w:val="0"/>')]
        if wide: out += [sect(2)] + blk + [sect(1)]
        else: out += blk
        prev = 'fig'
    elif tag == 'TABLE':
        blk, wide = table_xml(int(rest))
        if wide: out += [sect(2)] + blk + [sect(1)]
        else: out += blk
        prev = 'table'
    else:
        raise ValueError(line)

# references
out.append(para('H1', 'REFERENCES', '<w:keepNext/>'))
for k in order:
    out.append(para('References', C.REFS[k]))
unused = [k for k in C.REFS if k not in order]
assert not unused, unused
# biographies
out.append(para('H1', 'AUTHOR BIOGRAPHIES', '<w:keepNext/>'))
for b in C.BIOS:
    out.append(para('AUBios', b, '<w:spacing w:before="160"/>'))
out.append(para('FootnoteText', '{{Add a 1 in × 1.25 in head-and-shoulders photograph to the left of each biography, as in the IEEE Access template.}}', '<w:spacing w:before="120"/>'))

# ---------------- front matter ----------------
front = items[:12]
def replace_par_text(p, new_runs):
    ppr = re.search(r'<w:pPr>.*?</w:pPr>', p, flags=re.S)
    start = p[:p.find('>') + 1]
    return start + (ppr.group(0) if ppr else '') + ''.join(new_runs) + '</w:p>'
front[2] = replace_par_text(front[2], runs(C.TITLE))
front[3] = replace_par_text(front[3], runs(C.AUTHORS))
front[4] = replace_par_text(front[6], runs(C.AFFILS[0], '<w:sz w:val="14"/><w:szCs w:val="14"/>'))
front[5] = ''
front[6] = ''
front[7] = replace_par_text(front[7], runs(C.CORR))
front[8] = replace_par_text(front[8], runs(C.FUNDING))
abs_label = '<w:r><w:rPr><w:rStyle w:val="H5CharChar"/></w:rPr><w:t>ABSTRACT</w:t></w:r><w:r><w:t xml:space="preserve"> </w:t></w:r>'
front[9] = replace_par_text(front[9], [abs_label] + runs(C.ABSTRACT))
it_label = '<w:r><w:rPr><w:rStyle w:val="H5CharChar"/></w:rPr><w:t>INDEX TERMS</w:t></w:r><w:r><w:t xml:space="preserve"> </w:t></w:r>'
front[10] = replace_par_text(front[10], [it_label] + runs(C.INDEX_TERMS))

final_sect = final_sect.replace('r:id="rId25"', 'r:id="rId10"')
f1 = open(f'{OUT}/word/footer1.xml', encoding='utf8').read()
f1 = f1.replace('<w:tab/><w:t>1</w:t></w:r>', '<w:tab/></w:r><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>')
assert 'PAGE' in f1
open(f'{OUT}/word/footer1.xml', 'w', encoding='utf8').write(f1)
new = head + ''.join(front) + ''.join(out) + final_sect + '</w:body></w:document>'
open(f'{OUT}/word/document.xml', 'w', encoding='utf8').write(new)
open(f'{OUT}/word/_rels/document.xml.rels', 'w', encoding='utf8').write(rels)

# core properties
core = open(f'{OUT}/docProps/core.xml', encoding='utf8').read()
core = re.sub(r'<dc:title>.*?</dc:title>', f'<dc:title>{esc(C.TITLE)}</dc:title>', core)
core = re.sub(r'<dc:creator>.*?</dc:creator>', '<dc:creator>Authors</dc:creator>', core)
open(f'{OUT}/docProps/core.xml', 'w', encoding='utf8').write(core)

name = 'preview.docx' if os.environ.get('PREVIEW') else 'PTC_SSL_IEEE_Access_Manuscript.docx'
if os.path.exists(name): os.remove(name)
subprocess.run(f'cd {OUT} && zip -XrDq ../{name} .', shell=True, check=True)
print('refs order', order)
print('written', name)
