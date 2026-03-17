import React, { useState } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle, FileMusic, Image as ImageIcon } from 'lucide-react';
import './index.css';

function App() {
  const [formData, setFormData] = useState({
    email: '', name: '', roll_number: '', university: '',
    prev_gpa: '', prev_prev_gpa: '', midterm: '', assignment: '',
    twelfth: '', tenth: '', study_hours: '', attendance: '',
    backlogs: '', stress: '5', distance: '', complexity: '',
    teacher_fb: '', participation: ''
  });

  const [files, setFiles] = useState({ photo: null, audio: null });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleTextChange = (e) => {
    let { name, value } = e.target;
    // Strict numeric validations
    if (['midterm', 'assignment', 'twelfth', 'tenth', 'attendance'].includes(name)) {
      if (value > 100) value = 100;
      if (value < 0) value = 0;
    }
    if (['prev_gpa', 'prev_prev_gpa'].includes(name)) {
      if (value > 10) value = 10;
      if (value < 0) value = 0;
    }
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e) => {
    const { name, files: selectedFiles } = e.target;
    if (selectedFiles.length > 0) {
      setFiles(prev => ({ ...prev, [name]: selectedFiles[0] }));
    }
  };

  const submitData = async (e) => {
    e.preventDefault();
    setLoading(true);

    const data = new FormData();
    data.append('data', JSON.stringify(formData));
    if (files.photo) data.append('photo', files.photo);
    if (files.audio) data.append('audio', files.audio);

    try {
      await axios.post('http://localhost:3001/submit', data, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setSuccess(true);
      window.scrollTo(0, 0);
    } catch (err) {
      alert("Failed to submit. Is backend running?");
    }
    setLoading(false);
  };

  if (success) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="bg-white p-8 rounded-2xl shadow-xl max-w-md w-full text-center">
          <CheckCircle className="w-20 h-20 text-green-500 mx-auto mb-6" />
          <h2 className="text-3xl font-bold text-gray-800 mb-2">Thank You!</h2>
          <p className="text-gray-600">Your academic data has been successfully recorded for the research project.</p>
          <button 
            onClick={() => window.location.reload()}
            className="mt-8 bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition"
          >
            Submit Another Response
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-3xl mx-auto bg-white rounded-3xl shadow-xl overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-700 to-indigo-800 pt-16 pb-12 px-10">
          <h1 className="text-4xl font-extrabold text-white mb-4">Student Academic & Multi-Modal Dataset Collection</h1>
          <p className="text-blue-100 text-lg leading-relaxed">
            This secure portal collects academic, lifestyle, and multi-modal data for a university research project predicting CGPA using AI. Please enter your data accurately. All entries are restricted to valid numeric boundaries.
          </p>
        </div>

        {/* Form Body */}
        <form onSubmit={submitData} className="p-10 space-y-10">
          
          {/* Section 1: Demographics */}
          <div>
            <h3 className="text-xl font-bold text-gray-800 mb-6 border-b pb-2">1. Personal Details</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Email Address *</label>
                <input required type="email" name="email" value={formData.email} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 outline-none" placeholder="student@university.edu" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Full Name *</label>
                <input required type="text" name="name" value={formData.name} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 outline-none" placeholder="John Doe" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">University Roll Number *</label>
                <input required type="text" name="roll_number" value={formData.roll_number} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 outline-none" placeholder="e.g. 21BCE10234" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">University / College Name *</label>
                <input required type="text" name="university" value={formData.university} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 outline-none" placeholder="Harvard University" />
              </div>
            </div>
          </div>

          {/* Section 2: Core Academics */}
          <div>
            <h3 className="text-xl font-bold text-gray-800 mb-6 border-b pb-2">2. Academic History (Numbers Only)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Previous Semester CGPA *</label>
                <input required type="number" step="0.01" min="0" max="10" name="prev_gpa" value={formData.prev_gpa} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300" placeholder="0.00 to 10.00" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">CGPA of Last to Last Semester</label>
                <input type="number" step="0.01" min="0" max="10" name="prev_prev_gpa" value={formData.prev_prev_gpa} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300" placeholder="(Optional) 0.00 to 10.00" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Midterm Percentage *</label>
                <input required type="number" step="0.1" min="0" max="100" name="midterm" value={formData.midterm} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300" placeholder="0 to 100%" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Assignment Percentage *</label>
                <input required type="number" step="0.1" min="0" max="100" name="assignment" value={formData.assignment} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300" placeholder="0 to 100%" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">12th Grade Percentage *</label>
                <input required type="number" step="0.1" min="0" max="100" name="twelfth" value={formData.twelfth} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300" placeholder="0 to 100%" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">10th Grade Percentage *</label>
                <input required type="number" step="0.1" min="0" max="100" name="tenth" value={formData.tenth} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300" placeholder="0 to 100%" />
              </div>
            </div>
          </div>

          {/* Section 3: Lifestyle & Effort */}
          <div>
            <h3 className="text-xl font-bold text-gray-800 mb-6 border-b pb-2">3. Lifestyle & Behavioral Factors</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Study Hours Per Day *</label>
                <input required type="number" step="0.5" min="0" max="24" name="study_hours" value={formData.study_hours} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300" placeholder="e.g. 4.5" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Overall Attendance (%) *</label>
                <input required type="number" step="1" min="0" max="100" name="attendance" value={formData.attendance} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300" placeholder="0 to 100%" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Number of Backlogs *</label>
                <input required type="number" step="1" min="0" name="backlogs" value={formData.backlogs} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300" placeholder="0 if None" />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Distance From Campus (KM) *</label>
                <input required type="number" step="0.1" min="0" name="distance" value={formData.distance} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300" placeholder="0 if living on campus" />
              </div>
            </div>

            <div className="mt-8">
              <label className="block text-sm font-semibold text-gray-700 mb-4">Mental Stress Score (0: None, 10: Extreme) *</label>
              <div className="flex items-center space-x-4">
                <span className="text-sm font-bold text-green-600">0</span>
                <input type="range" name="stress" min="0" max="10" value={formData.stress} onChange={handleTextChange} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
                <span className="text-sm font-bold text-red-600">10</span>
              </div>
              <div className="text-center mt-2 font-bold text-blue-600 text-xl">{formData.stress}</div>
            </div>
          </div>

          {/* Section 4: Qualitative Interactions */}
          <div>
            <h3 className="text-xl font-bold text-gray-800 mb-6 border-b pb-2">4. Qualitative Classroom Interactions</h3>
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Complexity of Content (Presentations/Assignments) *</label>
                <select required name="complexity" value={formData.complexity} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300 bg-white">
                  <option value="">Select an option</option>
                  <option value="1">1 (Easy / Surface Level)</option>
                  <option value="2">2 (Medium / Standard)</option>
                  <option value="3">3 (Hard / Highly Technical)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Teacher's Feedback on Projects/Assignments *</label>
                <select required name="teacher_fb" value={formData.teacher_fb} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300 bg-white">
                  <option value="">Select an option</option>
                  <option value="1">1 (Poor / Needs Works)</option>
                  <option value="2">2 (Average / Expected)</option>
                  <option value="3">3 (Good / Exceptional)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Participation In Group Discussions *</label>
                <select required name="participation" value={formData.participation} onChange={handleTextChange} className="w-full px-4 py-3 rounded-lg border border-gray-300 bg-white">
                  <option value="">Select an option</option>
                  <option value="1">1 (Less Active)</option>
                  <option value="2">2 (Good Listener)</option>
                  <option value="3">3 (Shares Statistics/Brings Data)</option>
                  <option value="4">4 (Moderator/Leader)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Section 5: Multi-Modal Uploads */}
          <div>
            <h3 className="text-xl font-bold text-gray-800 mb-6 border-b pb-2">5. Multi-Modal Artificial Intelligence Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              
              {/* Audio Upload */}
              <div className={`p-6 border-2 border-dashed rounded-xl transition ${files.audio ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-bold text-gray-800 flex items-center">
                    <FileMusic className="w-5 h-5 mr-2 text-indigo-600" />
                    Audio Introduction *
                  </h4>
                </div>
                <p className="text-xs text-gray-500 mb-4">Record a short audio file (.mp3) introducing yourself, your goals, and hobbies in 5-8 sentences.</p>
                
                <input type="file" required accept="audio/*" name="audio" id="audio-upload" className="hidden" onChange={handleFileChange} />
                <label htmlFor="audio-upload" className="cursor-pointer block text-center py-3 bg-white border border-gray-300 shadow-sm rounded-lg hover:bg-gray-50 text-sm font-medium text-gray-700">
                  {files.audio ? files.audio.name : "Choose Audio File"}
                </label>
              </div>

              {/* Photo Upload */}
              <div className={`p-6 border-2 border-dashed rounded-xl transition ${files.photo ? 'border-pink-500 bg-pink-50' : 'border-gray-300 hover:border-gray-400'}`}>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-bold text-gray-800 flex items-center">
                    <ImageIcon className="w-5 h-5 mr-2 text-pink-600" />
                    Handwritten Notes *
                  </h4>
                </div>
                <p className="text-xs text-gray-500 mb-4">Upload a clear photo (.jpg, .png) of one page of your handwritten class notes.</p>
                
                <input type="file" required accept="image/*" name="photo" id="photo-upload" className="hidden" onChange={handleFileChange} />
                <label htmlFor="photo-upload" className="cursor-pointer block text-center py-3 bg-white border border-gray-300 shadow-sm rounded-lg hover:bg-gray-50 text-sm font-medium text-gray-700">
                  {files.photo ? files.photo.name : "Choose Image File"}
                </label>
              </div>

            </div>
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className={`w-full py-4 text-white text-lg font-bold rounded-xl flex items-center justify-center transition ${loading ? 'bg-blue-400 cursor-wait' : 'bg-blue-600 hover:bg-blue-700 shadow-lg'}`}
          >
            {loading ? "Saving Securely to Database..." : "Pledge Academic Data & Submit"}
            {!loading && <UploadCloud className="w-6 h-6 ml-2" />}
          </button>

        </form>
      </div>
    </div>
  );
}

export default App;
