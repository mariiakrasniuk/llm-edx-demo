const test=require('node:test');
const assert=require('node:assert/strict');
const T=require('../assets/transformer.js');
const tokens=[{id:976,text:'The'},{id:1001,text:' river'},{id:2323,text:' bank'}];
const vocabulary=[{id:1},{id:2},{id:3},{id:4},{id:5},{id:6},{id:7},{id:8}];
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);

test('softmax stays normalized for large logits and respects temperature',()=>{
 const p=T.softmax([1000,1001,1002]);close(p.reduce((a,b)=>a+b),1);assert.ok(p[2]>p[1]&&p[1]>p[0]);
 const concentrated=T.softmax([0,1,2],.2),flat=T.softmax([0,1,2],2);
 assert.ok(concentrated[2]>flat[2]);
});
test('a hand-computed row-vector transformation has the expected coordinates',()=>{
 assert.deepEqual(T.matmul([2,3],[[4,5],[6,7]]),[26,31]);
});
test('causal attention cannot use tokens in future positions',()=>{
 const a=T.forward(tokens,vocabulary),b=T.forward([...tokens.slice(0,2),{id:9999}],vocabulary);
 assert.deepEqual(a.rows[0].weights,[1,0,0]);assert.equal(a.rows[1].weights[2],0);
 assert.deepEqual(a.rows[0].output,b.rows[0].output);assert.deepEqual(a.rows[1].output,b.rows[1].output);
 for(const row of a.rows)close(row.weights.reduce((x,y)=>x+y),1);
});
test('a context edit preserves the last token lookup but changes its contextual output',()=>{
 const a=T.forward(tokens,vocabulary),b=T.forward([tokens[0],{id:8765,text:' money'},tokens[2]],vocabulary);
 assert.deepEqual(a.rows[2].embedding,b.rows[2].embedding);
 assert.deepEqual(a.rows[2].x,b.rows[2].x);
 assert.notDeepEqual(a.rows[2].afterAttention,b.rows[2].afterAttention);
 assert.notDeepEqual(a.probabilities,b.probabilities);
});
test('editing WQ[0,0] changes only query coordinate zero directly and affects downstream probabilities',()=>{
 const a=T.forward(tokens,vocabulary,.8),b=T.forward(tokens,vocabulary,-2);
 assert.deepEqual(a.rows[2].q.slice(1),b.rows[2].q.slice(1));
 close(b.rows[2].q[0]-a.rows[2].q[0],a.rows[2].normalized[0]*(-2-.8));
 assert.deepEqual(a.rows[2].k,b.rows[2].k);assert.deepEqual(a.rows[2].v,b.rows[2].v);
 assert.notDeepEqual(a.probabilities,b.probabilities);
});
test('each distribution covers the supplied vocabulary and normalized representations are finite',()=>{
 const result=T.forward(tokens,vocabulary);close(result.probabilities.reduce((x,y)=>x+y),1);
 assert.equal(result.probabilities.length,8);
 for(const row of result.rows){assert.equal(row.expanded.length,8);assert.ok(row.final.every(Number.isFinite));close(row.final.reduce((x,y)=>x+y),0);}
 assert.throws(()=>T.forward([],vocabulary));assert.throws(()=>T.softmax([1,2],0));
});
