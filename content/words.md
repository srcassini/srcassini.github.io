---
words: true
draft: false
---

{{< rawhtml >}} 
<h1 class="section-title main words"><a>Interesting words</a></h1> 
{{< /rawhtml >}} 

Words - many of them. I find some of these when reading books, papers, articles, or in converstation.

{{< rawhtml >}} 


<div id='shuffleButton' class="button shuffled">Test yourself with Shuffle Mode</div>


<table id='word_list' class='table'>
  <thead>
    <tr><th>Word</th><th>Definition</th> 
  </thead>
  <tbody id='word_body'></tbody>
</table>

<script type='module'>
  import { wordList } from '../words.js';

  function loadTableData(items) {
    const table = document.getElementById('word_body');

    wordList.forEach ( item => {
      let row = table.insertRow();
      let word = row.insertCell(0);
      word.innerHTML = item.word;
      let definition = row.insertCell(1);
      definition.innerHTML = item.definition;
      definition.classList.toggle('definition');
      definition.id = item.word;
    });
  }

  function compare( a, b ) {
    if ( a.word < b.word ){
      return -1;
    }
    if ( a.word > b.word ){
      return 1;
    }
    return 0;
  }

wordList.sort( compare );
loadTableData(wordList);

// Scramble word definitions

// create encoding of each definition 
function getEncodings(definitions) {
  var encodings = {};
  definitions.forEach (def => {
    var encoding = getEncoding(def);
    encodings[encoding] = def;
  })
  return encodings;
}

function getEncoding(def) {
  var encoding = new Array(26).fill(0);
  var counts = getCounts(def);
  for (const [char, count] of Object.entries(counts)) {
    encoding[char] = count;
  }
  return encoding.toString().replace(/[,]/g,'');
}

function getCounts(def) {
  var def = def.toLowerCase().replace(/[.!?\\-\s,]/g,'');
  var chars = {};
  let re = /ab+c/;
  for (var i = 0; i < def.length; i++) {
    var char = def.charCodeAt(i) - 97;
    if (chars[char]) {
      chars[char] ++;
    }
    else {
      chars[char] = 1;
    }
  }
  return chars;
}

function getShuffles(definitions) {
  var shuffles = [];
  definitions.forEach (def => {
    var shuffled = smartShuffle(def)
    shuffles.push(shuffled);
  })
  return shuffles;
}

function smartShuffle(def) {
  // Get indices of each character using regex pattern
  var charIndices = []
  for (var i = 0; i < def.length; i ++) {
    if (def[i].match(/([A-Za-z])/g)) {
      charIndices.push(i)
    }
  }
  var mapping = shuffle(charIndices)

  // Then shuffle the string while respecting punctuation and spaces
  var shuffledStr = new Array(def.length).fill('');
  let j = 0;
  for (var i = 0; i < def.length; i ++) {
    if (def[i].match(/([A-Za-z])/g)) {
      shuffledStr[mapping[j]] = def[i]
      j ++;
    }
    else {
      shuffledStr[i] = def[i];
    }
  }

  return shuffledStr.join('');
}

function shuffle(array) {
  let currentIndex = array.length,  randomIndex;

  // While there remain elements to shuffle...
  while (currentIndex != 0) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    // And swap it with the current element.
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
  }

  return array;
}

// Pre-encode definitions so shuffling can be as ENDLESS as you want
var words = [];
var wordDefinitions = [];
for (var i = 0; i < wordList.length; i++) {
  wordDefinitions.push(wordList[i].definition);
  words.push(wordList[i].word);
}
const allEncodings = getEncodings(wordDefinitions);

var toggleShuffle = false;

const shuffleButton = document.getElementById("shuffleButton");
const shuffleButtonText = shuffleButton.innerText;
shuffleButton.innerText = smartShuffle(shuffleButtonText);

const definitions = document.querySelectorAll('.definition');

shuffleButton.addEventListener("mouseenter", function( event ) {
  shuffleButton.innerText = shuffleButtonText;
})
shuffleButton.addEventListener("mouseleave", function( event ) {
  var text = shuffleButton.innerText;
  shuffleButton.innerText = smartShuffle(text);
})

shuffleButton.addEventListener("click", function (e) {
  // Shuffle and replace all definitions in the table
  toggleShuffle = !toggleShuffle

  if (toggleShuffle) {
    var shuffledDefinitions = getShuffles(wordDefinitions);
    for (var i = 0; i < words.length; i++) {
      var def = document.getElementById(words[i]);
      def.innerText = shuffledDefinitions[i];
    }
  }
  else {
    for (var i = 0; i < words.length; i++) {
      var def = document.getElementById(words[i]);
      var encoding = getEncoding(def.innerText);
      var original = allEncodings[encoding];
      def.innerText = original;
    }
  }
  
});


definitions.forEach (el => {
  var text = '';

  el.addEventListener("mouseenter", function( event ) {
    if (toggleShuffle) {
      // compare with original data to unscramble, i.e. search
      text = el.innerText;
      var encoding = getEncoding(text);
      var original = allEncodings[encoding];
      el.innerText = original;
    }
  })

  el.addEventListener("mouseleave", function( event ) {
    if (toggleShuffle) {
      el.innerText = smartShuffle(text);
    }
  })
})

</script>

{{< /rawhtml >}} 