function readURL(input){
    if(input.files && input.files[0]){
        let reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview').src = e.target.result;
        };
        reader.readAsDataURL(input.files[0]);
    }else{
        document.getElementById('preview').src = " ";
    }
}
    
function restPost_MNIST() {
    // FormData 객체 생성
    // FormData 객체 생성
    let formData = new FormData();

    // 이미지 파일을 가져오기 (예시: input 태그에 id가 'imageInput'일 때)
    let imageInput = document.getElementById('fileimg');
    let imageFile = imageInput.files[0];

    // FormData에 이미지 파일 추가
    formData.append('file', imageFile);

    console.log("post");
    console.log(formData);

    $.ajax({
        caches: false,
        url: "http://localhost:8080/MNIST",
        processData: false,
        contentType: false,
        type: 'POST',
        data: formData,
        beforeSend: function(xhr) {
            console.log("before");
        },
        success: function(response) {
            console.log("Success");
            let pnode1 = document.getElementById('resultP1');
            let pnode2 = document.getElementById('resultP2');
            
            console.log(response);

            const JP = JSON.stringify(response);
            const JPP = JSON.parse(JP);
            
            pnode1.innerHTML = `isSuccess : ${JPP.success}`;
            pnode2.innerHTML = `result : ${JPP.result}`;
        },
        error: function(xhr, status) {
            alert(xhr + " " + status);
        }
    });
}

function restPost_FMNIST() {
    // FormData 객체 생성
    let formData = new FormData();

    // 이미지 파일을 가져오기 (예시: input 태그에 id가 'imageInput'일 때)
    let imageInput = document.getElementById('fileimg');
    let imageFile = imageInput.files[0];

    // FormData에 이미지 파일 추가
    formData.append('file', imageFile);

    console.log("post");
    console.log(formData);

    $.ajax({
        caches: false, //ajax 캐시 처리
        url: "http://localhost:8080/FashionMNIST", // 호출 URL
        processData: false, // 기본값은 True이며, data 속성의 값이 콘텐츠 타입에 맞게 쿼리 문자열로 처리된다. 처리되지 않은 데이터를 보내려면 이 속성값을 false로 바꾸면 된다.
        contentType: false, // ajax를 통해서 서버에 데이터를 보낼 때 데이터 유형을 결정한다.
        type: 'POST', // http 타입 - REST API
        data: formData, // URL 호출시 보낼 파라미터 데이터
        beforeSend: function(xhr) { // API가 수행되기 전에 실행되는 함수
            console.log("before");
        },
        success: function(response) { // API가 성공적으로 수행되었을때 실행되는 함수
            console.log("Success");
            let pnode1 = document.getElementById('resultP1');
            let pnode2 = document.getElementById('resultP2');
            
            console.log(response);

            const JP = JSON.stringify(response);
            const JPP = JSON.parse(JP);
            
            pnode1.innerHTML = `isSuccess : ${JPP.success}`;
            pnode2.innerHTML = `result : ${JPP.result}`;
        }, 
        error: function(xhr, status) { // API가 에러가 발생했을때 실행되는 함수
            alert(xhr + " " + status);
        }
    });
}



