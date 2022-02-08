---
title: "Welcome!"
title_page: true
draft: false
---

Welcome to my website, where you can expect to find blogposts and interesting things, puzzles, algorithms, book reviews, experiences and career-building goodness. I am in my final masters year studying AI and Computer Science at the University of Sheffield, UK. Currently what interests me is chaos theory and trying to predict nonlinear behaviour - a seemingly incompatible statement - which falls under the umbrella of signal processing and AI. My masters thesis, run by [Dr. Joab Winkler](https://www.sheffield.ac.uk/dcs/people/academic/joab-winkler), involves the mathematical analysis of echo-state networks tasked with chaotic time series forecasting.

This, and (soon!) much more, you can read about in posts on this site. Check out my [latest blog](/posts/hexagon) on using an evolutionary algorithm to solve the magic hexagon problem.

{{< rawhtml >}}

<div class='button holder'>

  <div id='first' class="button link">LinkedIn</div>
  <div id='second' class="button link">CV</div> 
  <div id='third' class="button link">GitHub</div>

</div>

<script>
	var linkedin = document.getElementById('first');
	var cv = document.getElementById('second');
	var github = document.getElementById('third');

	// var toot = document.getElementsByTagName('a')[0];

	// toot.onclick = function (){
	// 		var audio = new Audio('toot.m4a');
	// 		audio.play();
	// }

	linkedin.addEventListener("mouseenter", function( event ) {
			let xhr = new XMLHttpRequest();
					xhr.open('GET', 'linkedin.m4a');
					xhr.responseType = 'arraybuffer';
			var audio = new Audio('linkedin.m4a');
			audio.play();
	})

	cv.addEventListener("mouseenter", function( event ) {
			let xhr = new XMLHttpRequest();
					xhr.open('GET', 'audio-CV.m4a');
					xhr.responseType = 'arraybuffer';
			var audio = new Audio('CV.m4a');
			audio.play();
	})

	github.addEventListener("mouseenter", function( event ) {
			let xhr = new XMLHttpRequest();
					xhr.open('GET', 'audio-github.m4a');
					xhr.responseType = 'arraybuffer';
			var audio = new Audio('github.m4a');
			audio.play();
	})

	linkedin.onclick = function (){
			window.open('https://www.linkedin.com/in/s-cassini/', '_blank');
			// var audio = new Audio('toot.m4a');
			// audio.play();
	}

	cv.onclick = function (){
			window.open('https://www.google.com/', '_blank');
			// var audio = new Audio('toot2.m4a');
			// audio.play();
	}

	github.onclick = function () {
			window.open('https://github.com/shauncassini', '_blank');
			// var audio = new Audio('toot3.m4a');
			// audio.play();
	}
</script>
{{< /rawhtml >}}

