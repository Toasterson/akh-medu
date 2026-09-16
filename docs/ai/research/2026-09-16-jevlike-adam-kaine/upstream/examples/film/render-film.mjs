import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const require=createRequire(import.meta.url);
const {chromium}=require('playwright');
const out=path.resolve(root,process.argv[2]||'videos/jevre-doom-10s.mp4');
const browser=await chromium.launch({headless:true});
try{
 const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});
 await page.goto('file://'+path.join(root,'replay.html')+'?capture');
 await page.evaluate(()=>window.ready);
 const ffmpeg=spawn('ffmpeg',['-y','-loglevel','error','-f','image2pipe','-vcodec','png','-r','24','-i','pipe:0','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',out],{stdio:['pipe','inherit','inherit']});
 const done=new Promise((yes,no)=>{ffmpeg.on('error',no);ffmpeg.on('exit',c=>c===0?yes():no(Error('ffmpeg '+c)))});
 const total=Math.round(24*await page.evaluate(()=>window.filmDuration));
 for(let frame=0;frame<total;frame++){await page.evaluate(t=>window.setFrame(t),frame/24);const png=await page.screenshot();if(!ffmpeg.stdin.write(png))await new Promise(r=>ffmpeg.stdin.once('drain',r));if(frame%96===0)console.log(`rendered ${frame}/${total}`)}
 console.log(`wrote ${out}`);
 ffmpeg.stdin.end();await done;
}finally{await browser.close()}
