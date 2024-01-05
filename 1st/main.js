//window.onload = function(){ } 함수는 웹 브라우저의 모든 구성요소에 대한 로드가 끝났을때, 브라우저에 의해서 호출되는 함수로,
//해당 부분에 넣으면, HTML을 모두 로드한 뒤에 함수를 호출합니다.
    window.onload = function(){
        const btnYellow = document.getElementById("button-yellow");
        const btnRed = document.getElementById("button-red");
        const btnBlue = document.getElementById("button-blue");
        const JSdiv = document.getElementById("JSbox");
        
        btnYellow.addEventListener("click", (e) => {
            JSdiv.style.backgroundColor = "yellow";
            console.log("Clicked Yellow!!");
        });

        btnRed.addEventListener("click", (e) => {
            JSdiv.style.backgroundColor = "red";
            console.log("Clicked Red!!");
        });

        btnBlue.addEventListener("click", (e) => {
            JSdiv.style.backgroundColor = "blue";
            console.log("Clicked Blue!!");
        });
    }