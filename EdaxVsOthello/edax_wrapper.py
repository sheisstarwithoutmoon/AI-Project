import subprocess
import time
import re

class EdaxWrapper:
    def __init__(self, edax_path='./edax.exe', depth=5):
        print("Starting Edax process...")
        args = [edax_path]
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            errors='replace'
        )
        time.sleep(1)
        stderr_output = self.process.stderr.readline()
        if stderr_output:
            print(f"Edax startup error: {stderr_output}")
        if self.process.poll() is not None:
            print("Edax process exited unexpectedly")
            raise RuntimeError("Failed to start Edax")
        self.send_cmd(f'setoption name SearchDepth value {depth}')

    def send_cmd(self, command):
        print(f"Sending command: {command}")
        try:
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError) as e:
            print(f"Error sending command: {e}")
            return False

    def get_move(self, fen):
        print(f"Sending FEN: {fen}")
        if not self.send_cmd(f'position fen {fen}'):
            return '0000'
        if not self.send_cmd('go'):
            return '0000'
        
        start_time = time.time()
        timeout = 10
        move_pattern = re.compile(r'[a-h][1-8]', re.IGNORECASE)
        
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline().strip()
            print(f"Edax output: {line}")
            
            # Look for 'bestmove' format output
            if line.startswith('bestmove'):
                move = line.split()[1].lower()
                print(f"Received bestmove: {move}")
                return move
                
            # Look for the 'Edax plays X' format
            if 'plays' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.lower() == 'plays' and i+1 < len(parts):
                        move = parts[i+1].lower()
                        print(f"Parsed move from 'plays': {move}")
                        if move_pattern.match(move):
                            return move
            
            # Look for a direct move in Edax book format
            if ' ' in line:
                fields = line.split()
                for field in fields:
                    if move_pattern.match(field.lower()):
                        print(f"Found potential move in book output: {field.lower()}")
                        return field.lower()
                        
            if not line or '>' in line:
                continue
                
            if 'error' in line.lower() or 'invalid' in line.lower():
                print("Edax reported an error")
                return '0000'
                
        print("Timeout waiting for Edax move")
        return '0000'

    def close(self):
        print("Closing Edax process")
        self.send_cmd('quit')
        try:
            self.process.terminate()
        except Exception as e:
            print(f"Error closing Edax: {e}")