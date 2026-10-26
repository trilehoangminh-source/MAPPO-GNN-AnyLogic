#!/usr/bin/env python3
"""Add bounded AGV17 movement-history instrumentation to a *current* AnyLogic .alp.

No modification to routing, transition conditions, node/path geometry or policies.
Requires an existing AGV_MOVEMENT_DEBUG block to ensure the input is the current
model, not the old baseline. Creates a separate output file; never overwrites.
"""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET

TRACE_MARKER = 'AGV17_TRACE_AUTO_PATCH_V1'
FUNCTION_NAME = 'dbgAgvMovement'


def trace_call(label: str) -> str:
    return f'{FUNCTION_NAME}("{label}");'


def inject_move_pair(action: str, label: str) -> str:
    pat = r'\bstop\(\);\s*moveTo\(decPos\);'
    replacement = '\n'.join([
        trace_call(label + '.beforeStop'),
        'stop();',
        trace_call(label + '.afterStop'),
        'moveTo(decPos);',
        trace_call(label + '.afterMoveTo'),
    ])
    action2, count = re.subn(pat, lambda _: replacement, action, count=1)
    if count != 1:
        raise RuntimeError(f'{label}: cannot locate stop(); moveTo(decPos); sequence')
    return action2


def patch_transition(element: str, name: str) -> str:
    match = re.search(r'(<Action>)(.*?)(</Action>)', element, re.S)
    if not match:
        raise RuntimeError(f'No Action on transition {name}')
    raw = match.group(2)
    has_cdata = raw.strip().startswith('<![CDATA[') and raw.strip().endswith(']]>')
    action = (raw.strip()[9:-3] if has_cdata else html.unescape(raw))
    if name in ('newNextNode', 'transition4'):
        action = inject_move_pair(action, name)
    elif name == 'transition3':
        action, n = re.subn(r'\bstop\(\);',
            lambda _: trace_call('transition3.beforeStop') + '\nstop();\n' + trace_call('transition3.afterStop'),
            action, count=1)
        if n != 1:
            raise RuntimeError('transition3: cannot locate stop()')
    elif name == 'transition7':
        if 'AGV_MOVEMENT_DEBUG' not in action:
            raise RuntimeError('Not the current Debug model: transition7 lacks AGV_MOVEMENT_DEBUG')
        # Keep original debug exception as nested cause, while adding a bounded
        # history of preceding movement commands.
        action = inject_move_pair(action, name)
        action, n = re.subn(r'\bstop\(\);\s*setXYZ\(\(Point\)\s*nodePos\);',
            lambda _: (trace_call('transition7.beforeSnap') + '\nstop();\n'
                       + trace_call('transition7.afterStopSnap') + '\n'
                       + 'setXYZ((Point) nodePos);\n'
                       + trace_call('transition7.afterSetXYZ')),
            action, count=1)
        if n != 1:
            raise RuntimeError('transition7: cannot locate stop(); setXYZ(nodePos);')
        action = '\n'.join([
            trace_call('transition7.beforeDistanceCheck'),
            'try {',
            action,
            '} catch (RuntimeException traceFailure) {',
            '  throw new IllegalStateException(',
            '    "AGV17_TRACE_FAILURE index=" + index + " time=" + time() +',
            '    "\\n=== RECENT MOVEMENT COMMANDS ===\\n" + debugMoveHistory,',
            '    traceFailure);',
            '}',
        ])
    else:
        raise RuntimeError('Unsupported transition ' + name)
    if has_cdata:
        if ']]>' in action:
            raise RuntimeError('Unexpected CDATA terminator inside Java Action')
        action_xml = '<![CDATA[' + action + ']]>'
    else:
        action_xml = html.escape(action, quote=False)
    return element[:match.start(2)] + action_xml + element[match.end(2):]


def patch_agv(agv: str, id1: int, id2: int) -> str:
    if TRACE_MARKER in agv or FUNCTION_NAME in agv:
        raise RuntimeError('Model appears already instrumented; refusing duplicate patch')
    variables = f'''<!-- {TRACE_MARKER} -->
                <Variable Class="PlainVariable">
                    <Id>{id1}</Id>
                    <Name>debugMoveHistory</Name>
                    <X>450</X><Y>60</Y>
                    <Label><X>10</X><Y>0</Y></Label>
                    <PublicFlag>false</PublicFlag>
                    <PresentationFlag>false</PresentationFlag>
                    <ShowLabel>false</ShowLabel>
                    <Properties SaveInSnapshot="false" Constant="false" AccessType="public" StaticVariable="false">
                        <Type><![CDATA[String]]></Type>
                        <InitialValue Class="CodeValue"><Code><![CDATA[""]]></Code></InitialValue>
                    </Properties>
                </Variable>
            '''
    if '</Variables>' not in agv:
        raise RuntimeError('AGV Variables section not found')
    agv = agv.replace('</Variables>', variables + '</Variables>', 1)
    fun_code = '''// Trace only the problem AGV, without ever calling getX()/getY()/distanceTo().
if (index != 17) return;
String entry =
    "[t=" + time() + "] " + label +
    " movingFlag=" + moving +
    " engineIsMoving=" + isMoving() +
    " engineTarget=(" + getTargetX() + "," + getTargetY() + ")" +
    " hasJumped=" + hasJumped +
    " currentNode=" + currentNode +
    " nextNode=" + nextNode +
    " lastNode=" + lastNode +
    " nodePos=" + nodePos +
    " decPos=" + decPos;
debugMoveHistory += entry + "\\n";
if (debugMoveHistory.length() > 14000)
    debugMoveHistory = debugMoveHistory.substring(debugMoveHistory.length() - 14000);'''
    func = f'''<Function AccessType="default" StaticFunction="false">
                    <ReturnModificator>VOID</ReturnModificator>
                    <ReturnType><![CDATA[double]]></ReturnType>
                    <Id>{id2}</Id>
                    <Name><![CDATA[{FUNCTION_NAME}]]></Name>
                    <X>450</X><Y>155</Y>
                    <Label><X>10</X><Y>0</Y></Label>
                    <PublicFlag>false</PublicFlag>
                    <PresentationFlag>false</PresentationFlag>
                    <ShowLabel>false</ShowLabel>
                    <Parameter>
                        <Name><![CDATA[label]]></Name>
                        <Type><![CDATA[String]]></Type>
                    </Parameter>
                    <Body><![CDATA[{fun_code}]]></Body>
                </Function>
            '''
    if '</Functions>' not in agv:
        raise RuntimeError('AGV Functions section not found')
    agv = agv.replace('</Functions>', func + '</Functions>', 1)
    done = {}
    def sub(m):
        element = m.group(0)
        n = re.search(r'<Name>(.*?)</Name>', element, re.S)
        name = re.sub(r'<!\[CDATA\[|\]\]>', '', n.group(1)) if n else ''
        if name in ('newNextNode', 'transition4', 'transition3', 'transition7'):
            if name in done:
                raise RuntimeError('Repeated transition ' + name)
            done[name] = True
            return patch_transition(element, name)
        return element
    agv = re.sub(r'<StatechartElement\b[^>]*>.*?</StatechartElement>', sub, agv, flags=re.S)
    required = {'newNextNode', 'transition4', 'transition3', 'transition7'}
    if set(done) != required:
        raise RuntimeError('Missing transitions: ' + repr(required - set(done)))
    return agv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path, help='Latest source .alp (NOT exported ZIP, NOT Matrix_old.alp)')
    p.add_argument('--output', type=Path, default=None, help='New file name, will not overwrite')
    a = p.parse_args()
    src = a.source.resolve()
    if not src.is_file() or src.suffix.lower() != '.alp':
        p.error('Provide the existing latest .alp file')
    dst = (a.output or src.with_name(src.stem + '_AGV17_trace.alp')).resolve()
    if dst == src or dst.exists():
        p.error('The destination must be a NEW file; no overwrite allowed')
    txt = src.read_text(encoding='utf-8-sig')
    agvs = list(re.finditer(r'^\t\t<ActiveObjectClass>\s*<Id>\d+</Id>.*?^\t\t</ActiveObjectClass>', txt, flags=re.S | re.M))
    matches = [m for m in agvs if re.search(r'<Name><!\[CDATA\[Agv\]\]></Name>', m.group(0)[:1000])]
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one ActiveObjectClass named Agv')
    ids = [int(i) for i in re.findall(r'<Id>(\d+)</Id>', txt)]
    new_ids = [max(ids) + 100, max(ids) + 101]
    m = matches[0]
    updated = patch_agv(m.group(0), *new_ids)
    out = txt[:m.start()] + updated + txt[m.end():]
    ET.fromstring(out)
    dst.write_text(out, encoding='utf-8')
    print('AGV17 TRACE PATCH READY')
    print('Source (unchanged):', src)
    print('Output:', dst)
    print('Instrumented: newNextNode, transition3, transition4, transition7')
    print('Next: open the OUTPUT .alp in AnyLogic, compile, export RL ZIP, replay episode 199.')

if __name__ == '__main__':
    main()
