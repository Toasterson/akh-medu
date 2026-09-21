// Builds replay.html for the film from a play trace.
//   node build-film.mjs ../../runs/doom-trace.json [seconds] [episode@start]
// seconds defaults to 5 (half of the 10 s film); episode@start (e.g. 3@12.0, seconds into
// that episode) takes a marked window, otherwise the busiest window is picked.
// The top band is one SVG in the diagram's own coordinate system: the bare dark
// tubeworks figure (background rectangle and the two open sock-end stubs removed)
// with a row of tensor heatmaps above it and a dotted leader from each heatmap to
// the point on the tube it reads from. film.html draws into the cells per frame.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const tracePath=process.argv[2];
if(!tracePath)throw Error('usage: node build-film.mjs TRACE.json [seconds] [episode@start]');
const trace=JSON.parse(await fs.readFile(path.resolve(process.cwd(),tracePath),'utf8'));
const seconds=Number(process.argv[3]||5);
if(!(seconds>0))throw Error('invalid seconds');
const windowLength=Math.round(seconds*35/4);
trace.film_seconds=seconds;
const marked=process.argv[4];
if(marked){
 const [episode,start]=marked.split('@').map(Number);
 const first=trace.decisions.findIndex(d=>d.episode===episode);
 if(first<0||!(start>=0))throw Error(`bad marked window ${marked}`);
 const begin=first+Math.round(start*35/4);
 if(begin+windowLength>trace.decisions.length||trace.decisions[begin+windowLength-1].episode!==episode)throw Error(`marked window ${marked} runs past episode ${episode}`);
 trace.selected_episode=episode;trace.selected_start_seconds=start;trace.selection_score=null;
 trace.decisions=trace.decisions.slice(begin,begin+windowLength);
}else if(trace.decisions.length>windowLength){
 let best={start:0,score:-Infinity};
 for(let start=0;start+windowLength<=trace.decisions.length;start++){
  const window=trace.decisions.slice(start,start+windowLength);
  if(window[0].episode!==window.at(-1).episode)continue;
  const score=window.reduce((total,decision,index)=>total
    +Math.max(0,decision.reward)*50
    +(decision.visible_objects||0)*1.5
    +(decision.action==='attack'?1:0)
    +(index&&decision.action!==window[index-1].action?2:0),0);   // liveliest window: kills, targets, firing, changes of mind
  if(score>best.score)best={start,score};
 }
 trace.selected_episode=trace.decisions[best.start].episode;
 trace.selected_start_seconds=best.start*4/35;
 trace.selection_score=best.score;
 trace.decisions=trace.decisions.slice(best.start,best.start+windowLength);
}
trace.kills=trace.decisions.filter(d=>d.reward>0).length;
trace.training_mean=trace.training_rewards.slice(-100).reduce((a,b)=>a+b,0)/Math.min(100,trace.training_rewards.length);
trace.probability_scale=Math.min(1,Math.max(...trace.decisions.flatMap(d=>d.probabilities))*1.08);

// --- the diagram, bare ---------------------------------------------------------
let svg=await fs.readFile(path.join(root,'diagram.svg'),'utf8');
const removed=[];
svg=svg.replace(/(<svg[^>]*>)<rect[^>]*fill="#0b0e14"\/>/,(m,open)=>{removed.push('background');return open});
for(const start of ['M 198\\.0 71\\.0 ','M 198\\.0 171\\.0 ']){
 svg=svg.replace(new RegExp(`<path d="${start}[^"]+" fill="none" stroke="#ff5f9e"[^>]*\\/>`),()=>{removed.push(start);return ''});
}
if(removed.length!==3)throw Error(`expected background and two stub strokes removed, got ${removed.join(', ')}`);

// --- heatmaps and leaders, in diagram units (1 unit = 960/792 px on the frame) ----
// Read points sit on the tube centrelines just after the plate each tensor leaves.
// One square cell for every tensor, so each footprint is its shape times the cell.
const U=6.6;   // cell edge for the matrices (8 px on the frame)
const first=trace.decisions[0].activations,N=trace.actions.length;
const CENTRE=-48.4; // every heatmap shares this vertical centre, above the topmost tube
const specs=[
 {id:'q',x:44, rows:N,cols:16,cw:U,  ch:U,  read:[102,48],  colour:'#4fc3f7'},
 {id:'k',x:164,rows:first.key.length,cols:16,cw:U,ch:U,read:[174,98],colour:'#4fc3f7'},
 {id:'v',x:284,rows:first.value.length,cols:16,cw:U,ch:U,read:[336,198],colour:'#ffc247'},
 {id:'a',x:404,rows:N,cols:40,cw:U,  ch:U,  read:[452,73],  colour:'#ff5f9e'},
 {id:'w',x:680,rows:N,cols:1, cw:U,  ch:U,  read:[646,135.5],colour:'#eef2f8'},
 {id:'p',x:720,rows:N,cols:1, cw:U,  ch:U,  read:[706,135.5],colour:'#eef2f8'},
];
function plate(s){
 const width=s.cols*s.cw,height=s.rows*s.ch,y=CENTRE-height/2;
 const cells=Array.from({length:s.rows*s.cols},(_,index)=>{
  const row=Math.floor(index/s.cols),column=index%s.cols;
  return `<rect data-cell="${index}" x="${(s.x+column*s.cw).toFixed(2)}" y="${(y+row*s.ch).toFixed(2)}" width="${(s.cw+.06).toFixed(2)}" height="${(s.ch+.06).toFixed(2)}" fill="#000"/>`;
 }).join('');
 const startX=Math.min(Math.max(s.read[0],s.x+s.cw/2),s.x+width-s.cw/2);
 const leader=`<line x1="${startX}" y1="${y+height+1.5}" x2="${s.read[0]}" y2="${s.read[1]}" stroke="#8a8f99" stroke-width="1.2" stroke-dasharray="1.6 3.2" stroke-linecap="round"/>`
  +`<circle cx="${s.read[0]}" cy="${s.read[1]}" r="2.6" fill="#d8dce3"/>`;
 return `<g id="cut-${s.id}" class="activation-cutaway">${cells}<rect x="${s.x-.6}" y="${y-.6}" width="${width+1.2}" height="${height+1.2}" fill="none" stroke="#6b7180" stroke-width="0.6"/>${leader}</g>`;
}
const top=CENTRE-Math.max(...specs.map(s=>s.rows*s.ch))/2-4;
const vb=/viewBox="([^"]+)"/.exec(svg)[1].split(' ').map(Number);
svg=svg.replace(/viewBox="[^"]+"/,`viewBox="${vb[0]} ${top} ${vb[2]} ${vb[1]+vb[3]-top}"`).replace(/ width="[^"]+" height="[^"]+"/,'');
svg=svg.replace('</svg>',`${specs.map(plate).join('')}</svg>`);
trace.diagram_viewbox=[vb[0],top,vb[2],vb[1]+vb[3]-top];

const template=await fs.readFile(path.join(root,'film.html'),'utf8');
const html=template.replace('/* DATA */',JSON.stringify(trace).replaceAll('</','<\\/')).replace('<!-- DIAGRAM -->',svg);
await fs.writeFile(path.join(root,'replay.html'),html);
console.log(`built ${trace.placeholder?'PLACEHOLDER ':''}${trace.decisions.length} measured decisions for ${seconds} s from episode ${trace.selected_episode||trace.decisions[0].episode} at ${(trace.selected_start_seconds||0).toFixed(2)} s; ${trace.kills} kills in window; probability scale ${trace.probability_scale.toFixed(2)}`);
