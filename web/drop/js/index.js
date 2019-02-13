// ************************ Drag and drop ***************** //
let dropArea = document.getElementById("drop-area")

// Prevent default drag behaviors
;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false)
    document.body.addEventListener(eventName, preventDefaults, false)
})

// Highlight drop area when item is dragged over it
;['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false)
})

;['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false)
})

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false)

function preventDefaults(e) {
    e.preventDefault()
    e.stopPropagation()
}

function highlight(e) {
    dropArea.classList.add('highlight')
}

function unhighlight(e) {
    dropArea.classList.remove('highlight')
}

function handleDrop(e) {
    var dt = e.dataTransfer
    var files = dt.files

    console.debug('drop')
    handleFiles(files)
}

let uploadProgress = []
let progressBar = document.getElementById('progress-bar')

function initializeProgress(numFiles) {
    progressBar.value = 0
    uploadProgress = []

    for (let i = numFiles; i > 0; i--) {
        uploadProgress.push(0)
    }
}

function updateProgress(fileNumber, percent) {
    uploadProgress[fileNumber] = percent
    let total = uploadProgress.reduce((tot, curr) => tot + curr, 0) / uploadProgress.length
    console.debug('update', fileNumber, percent, total)
    progressBar.value = total
}

function handleFiles(files) {
    console.debug('handle')
    files = [...files]
    console.log(files)
    initializeProgress(files.length)
    files.forEach(previewFile)
    files.forEach(uploadFile)
}

function previewFile(file) {
    let img_src = 'file.ico'
    switch (file.name.split('.').pop().toLowerCase()) {
        case 'text':
        case 'txt':
        case 'md':
            img_src = 'txt.ico'
            break
        case 'pdf':
            img_src = 'pdf.ico'
            break
    }
    let reader = new FileReader()
    console.debug('preview')
    reader.readAsDataURL(file)
    reader.onloadend = () => {
        let img = document.createElement('img')
        img.src = 'assets/' + img_src
        img.style.height = '30px'
        img.style.width = '30px'
        document.getElementById('gallery').appendChild(img)
        console.debug('load end')
    }
}

var modal = document.getElementById('folder-modal');

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks on <span> (x), close the modal
span.onclick = function () {
    modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function (event) {
    if (event.target === modal) {
        modal.style.display = "none";
    }
}

function radioButton(path, content) {
    var text = document.createElement('span')
    text.innerText = path
    content = content || text
    var input = document.createElement('input')
    var label = document.createElement('label')
    var div = document.createElement('div')
    input.setAttribute('type', 'radio')
    input.setAttribute('name', 'folder_choice')
    input.setAttribute('value', path)
    label.setAttribute("for", path)
    label.appendChild(content)
    div.appendChild(input)
    div.appendChild(label)
    return div
}

function uploadFile(file, i) {
    var url = 'http://127.0.0.1:8000/api/file'
    var xhr = new XMLHttpRequest()
    var formData = new FormData()
    xhr.open('POST', url, true)
    xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest')

    // Update progress (can be used to show progress indicator)
    xhr.upload.addEventListener("progress", function (e) {
        updateProgress(i, (e.loaded * 100.0 / e.total) || 100)
    })

    xhr.addEventListener('readystatechange', function (e) {
        if (xhr.readyState === 4 && xhr.status === 200) {
            updateProgress(i, 100) // <- Add this
            resp = JSON.parse(xhr.response)
            console.log(resp)
            note = document.getElementById('note')
            while (note.firstChild) {
                note.removeChild(note.firstChild);
            }
            var p = document.createElement("p")
            var strong = document.createElement('strong')
            var text = resp['input']
            strong.innerText = text.charAt(0).toUpperCase() + text.slice(1, 400) + '...'
            p.appendChild(strong)
            note.appendChild(p)
            resp['found'].forEach(function (text, i) {
                var p = document.createElement("p")
                p.innerText = '- [' + resp['score'][i].toString().slice(0, 5) + '] ' + text.charAt(0).toUpperCase() + text.slice(1, 400) + '...'
                note.appendChild(p)
            })
            modal.style.display = "block";
            var folderInputs = document.getElementById('folder-inputs')
            while (folderInputs.firstChild) {
                folderInputs.removeChild(folderInputs.firstChild);
            }
            resp['paths'].filter((value, index, array) => array.indexOf(value) === index).forEach(function (path) {
                folderInputs.appendChild(radioButton(path))
            })

            let input = document.createElement('input')
            input.type = "text"
            input.default = "Enter a new folder name"
            folderInputs.appendChild(radioButton('new_path', input))
        } else if (xhr.readyState === 4 && xhr.status !== 200) {
            alert('that did not work')
        }
    })
    formData.append('file', file)
    xhr.send(formData)
}
