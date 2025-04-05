import base64
import os
from module_solver_cs708 import *
from flask import Flask, request, jsonify

app = Flask(__name__)

# Store session data
sessions = {}

# Store chunks data
chunks_data = {}

@app.route("/", methods=['GET'])
def home():
    return "Hello, World!"

@app.route('/api/scene_image_chunk', methods=['POST'])
def upload_scene_image_chunk():
    if request.method == 'POST':
        try:
            # Get session ID and chunk info
            session_id = request.form.get('session_id')
            chunk_data = request.form.get('chunk_data')
            chunk_index = int(request.form.get('chunk_index'))
            total_chunks = int(request.form.get('total_chunks'))
            
            if not session_id or chunk_data is None or chunk_index is None or total_chunks is None:
                return jsonify({"error": "Missing required parameters"}), 400
                
            # Initialize chunks data for this session if it doesn't exist
            if session_id not in chunks_data:
                chunks_data[session_id] = {}
                
            # Initialize scene image chunks if they don't exist
            if 'scene_image' not in chunks_data[session_id]:
                chunks_data[session_id]['scene_image'] = {}
                chunks_data[session_id]['scene_image']['total_chunks'] = total_chunks
                chunks_data[session_id]['scene_image']['received_chunks'] = 0
                chunks_data[session_id]['scene_image']['chunks'] = {}
            
            # Store this chunk
            chunks_data[session_id]['scene_image']['chunks'][chunk_index] = chunk_data
            chunks_data[session_id]['scene_image']['received_chunks'] += 1
            
            # Return success response with progress information
            received = chunks_data[session_id]['scene_image']['received_chunks']
            total = chunks_data[session_id]['scene_image']['total_chunks']
            progress = int(received * 100 / total)
            
            return jsonify({
                "status": "success", 
                "message": f"Scene image chunk {chunk_index + 1}/{total_chunks} received",
                "progress": progress
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Method not allowed"}), 405

@app.route('/api/depth_data_chunk', methods=['POST'])
def upload_depth_image_chunk():
    if request.method == 'POST':
        try:
            # Get session ID and chunk info
            session_id = request.form.get('session_id')
            chunk_data = request.form.get('chunk_data')
            chunk_index = int(request.form.get('chunk_index'))
            total_chunks = int(request.form.get('total_chunks'))
            
            if not session_id or chunk_data is None or chunk_index is None or total_chunks is None:
                return jsonify({"error": "Missing required parameters"}), 400
                
            # Initialize chunks data for this session if it doesn't exist
            if session_id not in chunks_data:
                chunks_data[session_id] = {}
                
            # Initialize depth image chunks if they don't exist
            if 'depth_image' not in chunks_data[session_id]:
                chunks_data[session_id]['depth_image'] = {}
                chunks_data[session_id]['depth_image']['total_chunks'] = total_chunks
                chunks_data[session_id]['depth_image']['received_chunks'] = 0
                chunks_data[session_id]['depth_image']['chunks'] = {}
            
            # Store this chunk
            chunks_data[session_id]['depth_image']['chunks'][chunk_index] = chunk_data
            chunks_data[session_id]['depth_image']['received_chunks'] += 1
            
            # Return success response with progress information
            received = chunks_data[session_id]['depth_image']['received_chunks']
            total = chunks_data[session_id]['depth_image']['total_chunks']
            progress = int(received * 100 / total)
            
            return jsonify({
                "status": "success", 
                "message": f"Depth image chunk {chunk_index + 1}/{total_chunks} received",
                "progress": progress
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Method not allowed"}), 405

@app.route('/api/translation_matrix_chunk', methods=['POST'])
def upload_translation_matrix_chunk():
    if request.method == 'POST':
        try:
            # Get session ID and chunk info
            session_id = request.form.get('session_id')
            chunk_data = request.form.get('chunk_data')
            chunk_index = int(request.form.get('chunk_index'))
            total_chunks = int(request.form.get('total_chunks'))
            
            if not session_id or chunk_data is None or chunk_index is None or total_chunks is None:
                return jsonify({"error": "Missing required parameters"}), 400
                
            # Initialize chunks data for this session if it doesn't exist
            if session_id not in chunks_data:
                chunks_data[session_id] = {}
                
            # Initialize translation matrix chunks if they don't exist
            if 'translate_matrix' not in chunks_data[session_id]:
                chunks_data[session_id]['translate_matrix'] = {}
                chunks_data[session_id]['translate_matrix']['total_chunks'] = total_chunks
                chunks_data[session_id]['translate_matrix']['received_chunks'] = 0
                chunks_data[session_id]['translate_matrix']['chunks'] = {}
            
            # Store this chunk
            chunks_data[session_id]['translate_matrix']['chunks'][chunk_index] = chunk_data
            chunks_data[session_id]['translate_matrix']['received_chunks'] += 1
            
            # Return success response with progress information
            received = chunks_data[session_id]['translate_matrix']['received_chunks']
            total = chunks_data[session_id]['translate_matrix']['total_chunks']
            progress = int(received * 100 / total)
            
            return jsonify({
                "status": "success", 
                "message": f"Translation matrix chunk {chunk_index + 1}/{total_chunks} received",
                "progress": progress
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Method not allowed"}), 405

@app.route('/api/camera_matrix_chunk', methods=['POST'])
def upload_camera_matrix_chunk():
    if request.method == 'POST':
        try:
            # Get session ID and chunk info
            session_id = request.form.get('session_id')
            chunk_data = request.form.get('chunk_data')
            chunk_index = int(request.form.get('chunk_index'))
            total_chunks = int(request.form.get('total_chunks'))
            
            if not session_id or chunk_data is None or chunk_index is None or total_chunks is None:
                return jsonify({"error": "Missing required parameters"}), 400
                
            # Initialize chunks data for this session if it doesn't exist
            if session_id not in chunks_data:
                chunks_data[session_id] = {}
                
            # Initialize camera matrix chunks if they don't exist
            if 'camera_matrix' not in chunks_data[session_id]:
                chunks_data[session_id]['camera_matrix'] = {}
                chunks_data[session_id]['camera_matrix']['total_chunks'] = total_chunks
                chunks_data[session_id]['camera_matrix']['received_chunks'] = 0
                chunks_data[session_id]['camera_matrix']['chunks'] = {}
            
            # Store this chunk
            chunks_data[session_id]['camera_matrix']['chunks'][chunk_index] = chunk_data
            chunks_data[session_id]['camera_matrix']['received_chunks'] += 1
            
            # Return success response with progress information
            received = chunks_data[session_id]['camera_matrix']['received_chunks']
            total = chunks_data[session_id]['camera_matrix']['total_chunks']
            progress = int(received * 100 / total)
            
            return jsonify({
                "status": "success", 
                "message": f"Camera matrix chunk {chunk_index + 1}/{total_chunks} received",
                "progress": progress
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Method not allowed"}), 405

@app.route('/api/finalize_chunked_upload', methods=['POST'])
def finalize_chunked_upload():
    if request.method == 'POST':
        try:
            # Get session ID and data type
            session_id = request.form.get('session_id')
            data_type = request.form.get('data_type')
            
            if not session_id or not data_type:
                return jsonify({"error": "Missing required parameters"}), 400
                
            # Check if we have chunks for this session and data type
            if session_id not in chunks_data or data_type not in chunks_data[session_id]:
                return jsonify({"error": f"No chunks found for {data_type}"}), 400
                
            # Check if we have all chunks
            chunk_info = chunks_data[session_id][data_type]
            if chunk_info['received_chunks'] < chunk_info['total_chunks']:
                return jsonify({
                    "error": f"Not all chunks received. Got {chunk_info['received_chunks']} out of {chunk_info['total_chunks']}"
                }), 400
            
            # Initialize session if it doesn't exist
            if session_id not in sessions:
                sessions[session_id] = {}    
                
            # Combine all chunks in order
            combined_data = ""
            for i in range(chunk_info['total_chunks']):
                if i in chunk_info['chunks']:
                    combined_data += chunk_info['chunks'][i]
                else:
                    return jsonify({"error": f"Missing chunk {i}"}), 400
            
            # Save the combined data based on data type
            if data_type == 'scene_image':
                # For scene image: decode base64 and save as image file
                try:
                    # Remove potential Base64 prefix
                    if "base64," in combined_data:
                        combined_data = combined_data.split("base64,")[1]
                        
                    # Decode Base64 to binary
                    image_data = base64.b64decode(combined_data)
                    
                    # Save to file
                    file_path = f"./images/image/scene_environment.png"  # Use appropriate extension
                    with open(file_path, 'wb') as file:
                        file.write(image_data)
                        
                    sessions[session_id]['image_path'] = file_path
                    
                except Exception as e:
                    return jsonify({"error": f"Error processing image data: {str(e)}"}), 500
                
            elif data_type == 'depth_image':
                file_path = f"./images/depth/depth0.txt"
                with open(file_path, 'w') as file:
                    for line in combined_data.split("\n"):
                        if len(line) > 0:
                            file.write(f"{line}\n")
                sessions[session_id]['depth_image_path'] = file_path
                
            elif data_type == 'translate_matrix':
                file_path = f"./images/trans_matrix/matrix0.txt"
                with open(file_path, 'w') as file:
                    for line in combined_data.split("\n"):
                        if len(line) > 0:
                            file.write(f"{line}\n")
                sessions[session_id]['translate_matrix_path'] = file_path
                
            elif data_type == 'camera_matrix':
                file_path = f"./images/intric_matrix/matrix0.txt"
                with open(file_path, 'w') as file:
                    for line in combined_data.split("\n"):
                        if len(line) > 0:
                            file.write(f"{line}\n")
                sessions[session_id]['camera_matrix_path'] = file_path
            
            # Cleanup chunk data
            del chunks_data[session_id][data_type]
            
            return jsonify({
                "status": "success", 
                "message": f"{data_type} chunks combined successfully"
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Method not allowed"}), 405

@app.route('/api/command', methods=['POST'])
def send_command():
    if request.method == 'POST':
        try:
            # Get session ID
            session_id = request.form.get('session_id')
            if not session_id:
                return jsonify({"error": "No session_id provided"}), 400
                
            # Initialize session if it doesn't exist
            if session_id not in sessions:
                sessions[session_id] = {}
                
            # Get command data
            command = request.form.get('command')
            if not command:
                return jsonify({"error": "No command provided"}), 400
                
            # Save command data
            file_path = f"./images/command/command0.txt"
            with open(file_path, 'w') as file:
                file.write(command)
            
            # Store file path in session
            sessions[session_id]['command_path'] = file_path
            
            return jsonify({"status": "success", "message": "Command sent successfully"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Method not allowed"}), 405

@app.route('/api/process_target', methods=['POST'])
def process_target():
    if request.method == 'POST':
        try:
            # Get session ID
            session_id = request.form.get('session_id')
            if not session_id:
                return jsonify({"error": "No session_id provided"}), 400
                
            # Check if session exists
            if session_id not in sessions:
                return jsonify({"error": "Invalid session_id"}), 400
                
            # Check if all required data is present
            session_data = sessions[session_id]
            required_keys = ['image_path', 'depth_image_path', 'translate_matrix_path', 'camera_matrix_path', 'command_path']
            
            for key in required_keys:
                if key not in session_data:
                    return jsonify({"error": f"Missing required data: {key}"}), 400
            
            # Process data for target detection
            net = init_seg_model()
            output = module_sol(net)
            
            # Clean up session data
            del sessions[session_id]
            
            # Also clean up any remaining chunk data
            if session_id in chunks_data:
                del chunks_data[session_id]
            
            return str(output)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Method not allowed"}), 405

@app.route('/quit')
def _quit():
    os._exit(0)

if __name__ == "__main__":
    # Ensure all required directories exist
    os.makedirs("./images/image", exist_ok=True)
    os.makedirs("./images/depth", exist_ok=True)
    os.makedirs("./images/trans_matrix", exist_ok=True)
    os.makedirs("./images/intric_matrix", exist_ok=True)
    os.makedirs("./images/command", exist_ok=True)
    
    # Increase the maximum request size (in bytes)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
    
    app.run(host='0.0.0.0', port=5000)