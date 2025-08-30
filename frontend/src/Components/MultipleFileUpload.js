// src/components/MultipleFileUpload.js
import React, { useState, useEffect, useMemo } from "react";
import Uppy from "@uppy/core";
import { Dashboard } from "@uppy/react";
import XHRUpload from "@uppy/xhr-upload";
import '@uppy/core/css/style.min.css';
import '@uppy/dashboard/css/style.min.css';
import { X, FileText, Upload } from "lucide-react";
import DragDrop from '@uppy/drag-drop';


const MAX_FILES = 4;
const MAX_FILE_SIZE_MB = 5; // 5 MB
const MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024;

export default function MultipleFileUpload({ setUploadedText, setUploadedPreview, setActiveInput, setError }) {
  const [uppy] = useState(() =>
    new Uppy({ id: "multiUpload", autoProceed: true, restrictions: { maxNumberOfFiles: 5, maxFileSize: 5000000 } })
      .use(XHRUpload, { endpoint: "http://localhost:8000/api/upload-files/", fieldName: "files[]" })
  )

  useEffect(() => {
    uppy.on("upload-success", (file, response) => {
      try {
        const textContent = response.body?.text_content || ""
        const combinedText = textContent
        setUploadedText(combinedText)
        setUploadedPreview(combinedText.split("\n").slice(0, 4).join("\n"))
        setActiveInput("file")
      } catch (err) {
        setError("Failed to process uploaded file content.")
        console.error(err)
      }
    })

    return () => uppy.destroy()
  }, [uppy])

  return (
    <div>
      <h2 className="text-xl font-bold mb-2">Upload Files</h2>
      <DragDrop uppy={uppy} note="Max 5 files, 5MB each" />
    </div>
  )
}