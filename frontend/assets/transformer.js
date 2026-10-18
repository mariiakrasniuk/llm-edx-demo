// A transparent, untrained teaching transformer: 4 dimensions, 1 causal head,
// 1 pre-normalized decoder block, and a small, explicitly supplied vocabulary.
// These weights are hand chosen. They are NOT weights or activations from GPT.
(function(root){
'use strict';
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const add=(a,b)=>a.map((x,i)=>x+b[i]);
const matmul=(x,w)=>w[0].map((_,j)=>x.reduce((s,v,i)=>s+v*w[i][j],0));
const softmax=(x,t=1)=>{if(!(t>0))throw new Error('Temperature must be positive.');const m=Math.max(...x);const e=x.map(v=>Math.exp((v-m)/t));const total=e.reduce((a,b)=>a+b,0);return e.map(v=>v/total);};
const norm=x=>{const mean=x.reduce((a,b)=>a+b,0)/x.length;const variance=x.reduce((s,v)=>s+(v-mean)**2,0)/x.length;return x.map(v=>(v-mean)/Math.sqrt(variance+1e-5));};
const gelu=x=>.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x**3)));
function embedding(id){let seed=(id+1)>>>0;return Array.from({length:4},()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed/4294967296)*2-1;});}
const position=i=>[Math.sin(i),Math.cos(i),Math.sin(i/10),Math.cos(i/10)].map(v=>v*.15);
const WQ=[[.8,.2,-.1,0],[-.2,.9,.1,.3],[.1,0,.7,-.2],[.3,-.1,.2,.8]];
const WK=[[.7,0,.3,-.1],[.2,.8,-.2,.1],[-.1,.3,.9,0],[0,-.2,.1,.7]];
const WV=[[.8,.1,0,-.2],[.2,.6,-.1,.1],[0,-.2,.9,.2],[-.1,.2,.1,.7]];
const WO=[[.6,.1,0,.2],[0,.7,-.1,.1],[.2,0,.6,-.1],[-.1,.2,.1,.8]];
const W1=[[.5,-.3,.8,0,.2,-.1,.4,.1],[.1,.6,-.2,.5,-.4,.7,.1,-.2],[-.3,.2,.1,.6,.8,0,-.5,.3],[.7,-.1,.4,-.2,.3,.5,.2,.6]];
const W2=[[.4,.1,-.2,.3],[-.3,.5,.1,0],[.2,-.2,.6,.1],[.1,.4,-.1,.5],[-.2,.3,.2,.4],[.5,-.1,.3,-.2],[.3,.2,-.4,.1],[-.1,.4,.2,.3]];
function forward(tokens,vocabulary,coefficient=.8,temperature=1){
 if(!tokens.length||tokens.length>32)throw new Error('The teaching model needs 1–32 tokens.');
 const wq=WQ.map(row=>[...row]);wq[0][0]=coefficient;
 const rows=tokens.map((token,i)=>{const e=embedding(token.id),p=position(i),x=add(e,p),normalized=norm(x);return {token,index:i,embedding:e,position:p,x,normalized,q:matmul(normalized,wq),k:matmul(normalized,WK),v:matmul(normalized,WV)};});
 for(const row of rows){row.scores=rows.map((other,j)=>j<=row.index?dot(row.q,other.k)/2:-Infinity);row.weights=softmax(row.scores);row.mixed=Array.from({length:4},(_,d)=>rows.reduce((sum,r,j)=>sum+row.weights[j]*r.v[d],0));row.attention=matmul(row.mixed,WO);row.afterAttention=add(row.x,row.attention);row.mlpInput=norm(row.afterAttention);row.expanded=matmul(row.mlpInput,W1);row.activated=row.expanded.map(gelu);row.mlp=matmul(row.activated,W2);row.output=add(row.afterAttention,row.mlp);row.final=norm(row.output);}
 const last=rows.at(-1);const logits=vocabulary.map(t=>dot(last.final,embedding(t.id))/2);const probabilities=softmax(logits,temperature);
 return {rows,logits,probabilities,wq,temperature,coefficient,vocabulary};
}
const api={dot,add,matmul,softmax,norm,gelu,embedding,position,forward,weights:{WQ,WK,WV,WO,W1,W2}};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.TeachingTransformer=api;
})(typeof window!=='undefined'?window:globalThis);
