import os
import uuid
import subprocess
import time
import re
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import nbformat # For interacting with .ipynb files
from jupyter_client.manager import KernelManager # For starting and managing a kernel
from jupyter_client.blocking import BlockingKernelClient # For communicating with the kernel
import queue # For non-blocking message retrieval
import datetime as dt
import requests
from enum import Enum
import lllm.utils as U
import atexit



KernelManager.transport = "ipc"


class JupyterCellType(Enum):
    MARKDOWN = 'markdown'
    CODE = 'code'

class ProgrammingLanguage(Enum):
    PYTHON = 'python' # now only support python
    # R = 'r' # TODO: maybe add support for R


@dataclass
class JupyterSession:
    """Represents a Jupyter Notebook Session, including its server process and kernel."""
    name: str
    dir: str # Path to the directory containing the notebook file
    metadata: Dict[str, Any]
    notebook_file: Optional[str] = None
    programming_language: ProgrammingLanguage = ProgrammingLanguage.PYTHON

    # For Jupyter Server process (launching the web UI)
    server_process: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    server_url: Optional[str] = field(default=None, init=False, repr=False)
    server_port: Optional[int] = field(default=None, init=False, repr=False)

    # For direct Kernel interaction (programmatic cell execution)
    kernel_manager: Optional[KernelManager] = field(default=None, init=False, repr=False)
    kernel_client: Optional[BlockingKernelClient] = field(default=None, init=False, repr=False)
    last_stop_index: int = 0 # if error, reset it
    _verbose: bool = False

    def silence(self):
        self._verbose = False

    def verbose(self):
        self._verbose = True

    def __post_init__(self):
        self.init_session()
        self.run_all_cells()

    def init_session(self):
        _cutoff_date = self.metadata['proxy']['cutoff_date']
        if _cutoff_date:
            if isinstance(_cutoff_date, str):
                _cutoff_date = dt.datetime.strptime(_cutoff_date, '%Y-%m-%d')
            assert isinstance(_cutoff_date, dt.datetime), f"Cutoff date must be a datetime object"
            _cutoff_date_str = f'\'{_cutoff_date.strftime('%Y-%m-%d')}\''
        else:
            _cutoff_date_str = 'None'
        _project_root = self.metadata['project_root']
        if os.name == 'nt':
            _project_root = _project_root.replace('\\', '/')
        _init_code = f'''# INIT CODE (DO NOT REMOVE THIS CELL)
import sys
sys.path.append('{_project_root}')
from analytica.proxy.re import Proxy
proxy = Proxy(activate_proxies={self.metadata['proxy']['activate_proxies']}, cutoff_date={_cutoff_date_str}, deploy_mode={self.metadata['proxy']['deploy_mode']})
CALL_API = proxy.__call__'''
        cell_0_content = self.cells[0].source if self.cells else ''
        if not cell_0_content.strip().startswith('# INIT CODE'):    
            self.insert_cell(0, _init_code, JupyterCellType.CODE)
        else:
            self.overwrite_cell(0, _init_code, JupyterCellType.CODE)

    def to_dict(self) -> Dict[str, Any]:
        _metadata = self.metadata.copy()
        _datetime = _metadata['proxy']['cutoff_date']
        if isinstance(_datetime, dt.datetime):
            _metadata['proxy']['cutoff_date'] = _datetime.strftime('%Y-%m-%d')
        elif isinstance(_datetime, str):
            _metadata['proxy']['cutoff_date'] = _datetime
        else:
            _metadata['proxy']['cutoff_date'] = None
        return {
            'name': self.name,
            'dir': self.dir,
            'metadata': _metadata,
            'notebook_file': self.notebook_file,
            'programming_language': self.programming_language.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JupyterSession':
        # Ensure notebook_file is absolute or resolved correctly if needed
        session_dir = data.get('dir')
        notebook_file = data.get('notebook_file')
        if session_dir and notebook_file and not os.path.isabs(notebook_file):
            notebook_file = U.pjoin(session_dir, os.path.basename(notebook_file))

        _metadata = data['metadata'].copy()
        _datetime = _metadata['proxy']['cutoff_date']
        if isinstance(_datetime, str):
            _metadata['proxy']['cutoff_date'] = dt.datetime.strptime(_datetime, '%Y-%m-%d')
        elif isinstance(_datetime, dt.datetime):
            _metadata['proxy']['cutoff_date'] = _datetime
        else:
            _metadata['proxy']['cutoff_date'] = None
        return cls(
            name=data['name'],
            dir=session_dir,
            metadata=_metadata,
            notebook_file=notebook_file,
            programming_language=ProgrammingLanguage(data['programming_language']),
        )


    def _read_notebook_object(self) -> Optional[nbformat.NotebookNode]:
        if not self.notebook_file or not U.pexists(self.notebook_file):
            # print(f"Notebook file {self.notebook_file} does not exist for reading.")
            return nbformat.v4.new_notebook() # Return empty notebook if file missing
        try:
            with open(self.notebook_file, 'r', encoding='utf-8') as f:
                return nbformat.read(f, as_version=4)
        except Exception as e:
            if self._verbose:
                print(f"Error reading notebook {self.notebook_file}: {e}")
            return nbformat.v4.new_notebook() # Return empty on error

    def _read_notebook_cells(self) -> List[nbformat.NotebookNode]:
        nb = self._read_notebook_object()
        return nb.cells if nb else []

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        return self._read_notebook_cells()
    
    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def get_cells(self, index: int | List[int]) -> List[nbformat.NotebookNode]:
        if isinstance(index, int):
            index = [index]
        return [self.cells[i] for i in index]

    @property
    def directory_tree(self) -> str:
        return U.directory_tree(self.dir)

    def _ensure_notebook_file(self, create: bool = True) -> None:
        if not self.notebook_file:
            self.notebook_file = U.pjoin(self.dir, f"{self.name}.ipynb")
        if create and not U.pexists(self.notebook_file):
            nb = nbformat.v4.new_notebook()
            try:
                with open(self.notebook_file, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                if self._verbose:
                    print(f"Created empty notebook: {self.notebook_file}")
            except Exception as e:
                if self._verbose:
                    print(f"Error creating notebook file {self.notebook_file}: {e}")
                self.notebook_file = None

    def _write_notebook_object(self, nb: nbformat.NotebookNode):
        if not self.notebook_file:
            if self._verbose:
                print("Error: Notebook file path is not set. Cannot write.")
            return
        try:
            with open(self.notebook_file, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
        except Exception as e:
            if self._verbose:
                print(f"Error writing to notebook file {self.notebook_file}: {e}")
            
    def _write_cell(self, content: str, cell_type: JupyterCellType,
                    ensure_exists: bool = True,
                    overwrite_index: Optional[int] = None,
                    insert_index: Optional[int] = None) -> int: # return the index of the cell
        self._ensure_notebook_file(create=ensure_exists)
        if not self.notebook_file:
            raise ValueError("Error: Cannot write cell, notebook file not set or couldn't be created.")
        
        assert overwrite_index is None or insert_index is None, "Cannot specify both overwrite_index and insert_index"

        nb = self._read_notebook_object()
        if not nb: 
            nb = nbformat.v4.new_notebook() # Should be handled by _read_notebook_object but defensive

        if cell_type == JupyterCellType.MARKDOWN:
            new_cell = nbformat.v4.new_markdown_cell(content)
        elif cell_type == JupyterCellType.CODE:
            new_cell = nbformat.v4.new_code_cell(content)
        else:
            raise ValueError(f"Invalid cell type: {cell_type}")
    
        if overwrite_index is not None:
            if 0 <= overwrite_index < len(nb.cells):
                nb.cells[overwrite_index] = new_cell
            else:
                raise ValueError(f"Error: overwrite_index {overwrite_index} out of bounds for {len(nb.cells)} cells.")
        elif insert_index is not None:
            if 0 <= insert_index <= len(nb.cells):
                nb.cells.insert(insert_index, new_cell)
            else:
                raise ValueError(f"Error: insert_index {insert_index} out of bounds for {len(nb.cells)} cells.")
        else:
            nb.cells.append(new_cell)

        self._write_notebook_object(nb)
        if self._verbose:
            print(f"Modified {cell_type.value} cell in {self.notebook_file}")
        return len(nb.cells) - 1
    
    def append_code_cell(self, content: str, ensure_exists: bool = True) -> int:
        return self._write_cell(content, JupyterCellType.CODE, ensure_exists)
    
    def append_markdown_cell(self, content: str, ensure_exists: bool = True) -> int:
        return self._write_cell(content, JupyterCellType.MARKDOWN, ensure_exists)

    def overwrite_cell(self, index: int, content: str, cell_type: JupyterCellType):
        self._write_cell(content, cell_type, ensure_exists=True, overwrite_index=index)

    def insert_cell(self, index: int, content: str, cell_type: JupyterCellType):
        self._write_cell(content, cell_type, ensure_exists=True, insert_index=index)

    def delete_cells(self, index: int | List[int]):
        if isinstance(index, int):
            index = [index]
        to_delete = []
        nb = self._read_notebook_object()
        for i in index:
            if not nb or not (0 <= i < len(nb.cells)):
                raise ValueError(f"Error: Cannot delete cell at index {i}. Notebook/cell not found or index out of bounds.")
            to_delete.append(nb.cells[i])
        for cell in to_delete:
            nb.cells.remove(cell)
        self._write_notebook_object(nb)
        if self._verbose:
            print(f"Deleted cells at indices {index} from {self.notebook_file}")


    # --- Jupyter Server (Web UI) Methods ---
    def launch_server(self, specific_port: Optional[int] = None) -> Optional[str]:
        if self.server_process and self.server_process.poll() is None:
            if self._verbose:
                print(f"Server for session '{self.name}' already running. URL: {self.server_url}")
            return self.server_url

        self._ensure_notebook_file(create=True)
        command = ['jupyter', 'notebook', f'--notebook-dir={self.dir}', '--no-browser', '--ip=127.0.0.1']
        if specific_port is not None:
            command.append(f'--port={specific_port}')
        else:
            command.append('--port-retries=50')

        if self._verbose:
            print(f"Launching Jupyter server for session '{self.name}' in '{self.dir}'...")
        try:
            self.server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
            url_pattern = re.compile(r'(http://(127\.0\.0\.1|localhost):(\d+)/\?token=\w+)')
            start_time = time.time()
            timeout_seconds = 20
            
            stderr_output = ""
            while time.time() - start_time < timeout_seconds:
                if self.server_process.stderr:
                    line = self.server_process.stderr.readline()
                    if line:
                        stderr_output += line
                        # print(f"[JupyterServer stderr] {line.strip()}") # Can be very verbose
                        match = url_pattern.search(line)
                        if match:
                            self.server_url = match.group(1)
                            if self.notebook_file and U.pexists(self.notebook_file):
                                nb_filename = os.path.basename(self.notebook_file)
                                encoded_nb_filename = requests.utils.quote(nb_filename)
                                self.server_url = f"{self.server_url.split('?')[0]}notebooks/{encoded_nb_filename}?{self.server_url.split('?')[1]}"
                            
                            port_match = re.search(r':(\d+)/', self.server_url)
                            if port_match: self.server_port = int(port_match.group(1))
                            if self._verbose:
                                print(f"Jupyter server started. Access URL: {self.server_url}, PID: {self.server_process.pid}")
                            return self.server_url
                    elif self.server_process.poll() is not None:
                        if self._verbose:
                            print(f"Jupyter server process terminated unexpectedly (exit code: {self.server_process.poll()}).")
                        break
                    time.sleep(0.1)
                else: time.sleep(0.2)
            
            if self._verbose:
                print(f"Error: Could not find Jupyter server URL in stderr within {timeout_seconds}s.")
            # print("--- Full stderr from server --- \n", stderr_output) # Only print if debugging
            self.shutdown_server()
            return None
        except FileNotFoundError:
            if self._verbose:
                print("Error: 'jupyter' command not found for server.")
            self.server_process = None
            return None
        except Exception as e:
            if self._verbose:
                print(f"An unexpected error occurred during server launch: {e}")
            if self.server_process: self.shutdown_server()
            return None

    def shutdown_server(self):
        if self.server_process and self.server_process.poll() is None:
            print(f"Shutting down Jupyter server for session '{self.name}' (PID: {self.server_process.pid})...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill(); self.server_process.wait()
            finally:
                if self.server_process.stdout: self.server_process.stdout.close()
                if self.server_process.stderr: self.server_process.stderr.close()
            print("Jupyter server shut down.")
        self.server_process, self.server_url, self.server_port = None, None, None

    # --- Kernel Interaction Methods ---
    def start_kernel(self) -> bool:
        if self.kernel_manager and self.kernel_manager.is_alive():
            # print("Kernel already running for this session.")
            return True
        

        lock = U.make_file_lock('analytica_jupyter_kernel', timeout=20)

        if self._verbose:
            print(f"Starting kernel for session '{self.name}'...")
        try:
            with lock:
                self.kernel_manager = KernelManager(kernel_name='analytica', env=os.environ) # Assumes ipykernel
                self.kernel_manager.start_kernel()
                self.kernel_client = self.kernel_manager.client()
                self.kernel_client.start_channels()
                try:
                    self.kernel_client.wait_for_ready(timeout=10)
                    # time.sleep(0.1)
                    if self._verbose:
                        print(f"Kernel started and ready (ID: {self.kernel_manager.kernel_id}).")
                    self.last_stop_index = 0 # kernel restart, reset it
                    return True
                except RuntimeError:
                    if self._verbose:
                        print("Timeout waiting for kernel to become ready."); self.shutdown_kernel(); return False
        except TimeoutError:
            if self._verbose:
                print(f"Failed to acquire lock on analytica_jupyter_kernel, another process may be holding it.")
                self.shutdown_kernel(); return False
        except Exception as e:
            if self._verbose:
                print(f"Failed to start kernel: {e}"); self.shutdown_kernel(); return False

    def shutdown_kernel(self):
        client_stopped, manager_stopped = False, False
        if self.kernel_client:
            try: self.kernel_client.stop_channels(); client_stopped = True
            except Exception as e:
                if self._verbose:
                    print(f"Error stopping client channels: {e}")
        if self.kernel_manager and self.kernel_manager.is_alive():
            try: self.kernel_manager.shutdown_kernel(now=True); manager_stopped = True
            except Exception as e:
                if self._verbose:
                    print(f"Error shutting down kernel: {e}")
        elif self.kernel_manager: # Exists but not alive
            manager_stopped = True # Effectively
        
        self.kernel_client, self.kernel_manager = None, None
        if client_stopped or manager_stopped:
            if self._verbose:
                print("Kernel resources released.")
        self.last_stop_index = 0 # kernel restart, reset it

    def run_cell(self, index: int, timeout: int = 60) -> bool:
        if not self.start_kernel():
            if self._verbose:
                print(f"Cannot run cell {index}: Kernel failed to start.")
            return False

        nb = self._read_notebook_object()
        if not nb or not (0 <= index < len(nb.cells)):
            if self._verbose:
                print(f"Error: Cell index {index} out of bounds or notebook not found.")
            return False
        
        cell_to_run = nb.cells[index]
        if cell_to_run.cell_type != 'code':
            return True 

        code = cell_to_run.source
        if not code.strip():
            cell_to_run.outputs = [] 
            self._write_notebook_object(nb)
            return True

        # This print is already in user's logs:
        # print(f"\nExecuting cell {index} in kernel for session '{self.name}':\n---\n{code}\n---")
        msg_id = self.kernel_client.execute(code, store_history=True)
        cell_to_run.outputs = [] 
        
        # Loop to gather IOPub messages
        # This loop continues as long as the kernel is not idle for this specific request,
        # or until the main timeout for shell reply is approaching.
        stop_iopub_gathering_loop = False
        # Use monotonic time for robust timeout calculation
        # This timeout is for the iopub gathering loop itself, ensure it doesn't block indefinitely
        # if the shell reply is somehow missed or the 'status:idle' message doesn't arrive.
        # The final get_shell_msg has its own timeout.
        iopub_loop_start_time = time.monotonic()
        # Give at least 1-2 seconds for the final shell reply after this loop
        max_iopub_loop_duration = timeout - 2 


        while not stop_iopub_gathering_loop:
            current_loop_duration = time.monotonic() - iopub_loop_start_time
            if current_loop_duration > max_iopub_loop_duration and max_iopub_loop_duration > 0 :
                # print(f"[DEBUG] Cell {index}: IOPub loop approaching overall cell timeout. Stopping iopub gathering.")
                break

            try:
                # Poll iopub with a short timeout to remain responsive
                msg = self.kernel_client.get_iopub_msg(timeout=0.2) 
                
                if msg['parent_header'].get('msg_id') == msg_id:
                    # print(f"[DEBUG] Cell {index} IOPUB: {msg['header']['msg_type']}") # Very verbose
                    try:
                        output = nbformat.v4.output_from_msg(msg)
                        cell_to_run.outputs.append(output)
                    except ValueError: 
                        pass 
                    
                    if msg['header']['msg_type'] == 'status' and \
                       msg['content']['execution_state'] == 'idle':
                        # print(f"[DEBUG] Cell {index}: IOPUB status idle received. Stopping iopub gathering.")
                        stop_iopub_gathering_loop = True
            
            except queue.Empty:
                # IOPub is quiet for the poll_interval (0.2s).
                # Check if the definitive shell reply has arrived.
                try:
                    shell_check = self.kernel_client.get_shell_msg(timeout=0) # Non-blocking check
                    if shell_check and shell_check['parent_header'].get('msg_id') == msg_id:
                        # print(f"[DEBUG] Cell {index}: Shell reply found during iopub quiet. Stopping iopub gathering.")
                        stop_iopub_gathering_loop = True # Shell reply means execution is done
                except queue.Empty:
                    # Shell is also quiet. Continue polling IOPub.
                    # If kernel is genuinely stuck or very slow, the max_iopub_loop_duration will eventually break this.
                    pass 
            
            except Exception as e:
                if self._verbose:
                    print(f"Error processing iopub message for cell {index}: {e}")
                stop_iopub_gathering_loop = True # Break on other errors

        execution_successful = False
        remaining_timeout_for_shell = max(1, timeout - (time.monotonic() - iopub_loop_start_time))

        try:
            # print(f"[DEBUG] Cell {index}: Exited iopub loop. Getting final shell reply with timeout {remaining_timeout_for_shell:.2f}s.")
            shell_reply = self.kernel_client.get_shell_msg(timeout=remaining_timeout_for_shell)
            if shell_reply['parent_header'].get('msg_id') == msg_id:
                status = shell_reply['content']['status']
                if self._verbose:
                    print(f"Kernel execution status for cell {index}: {status}") # This is a useful print
                if status == 'ok':
                    cell_to_run.execution_count = shell_reply['content'].get('execution_count')
                    execution_successful = True
                elif status == 'error':
                    # Error details should have been captured by output_from_msg from iopub.
                    # This is a fallback if iopub didn't send a typical error structure.
                    if not any(out.output_type == 'error' for out in cell_to_run.outputs):
                        err_output = nbformat.v4.new_output(
                            output_type='error',
                            ename=shell_reply['content']['ename'],
                            evalue=shell_reply['content']['evalue'],
                            traceback=shell_reply['content']['traceback']
                        )
                        cell_to_run.outputs.append(err_output)
            else:
                if self._verbose:
                    print(f"Warning: Mismatched shell reply for cell {index}.")
                if not any(out.output_type == 'error' for out in cell_to_run.outputs):
                    cell_to_run.outputs.append(nbformat.v4.new_output(output_type='error', ename='ShellReplyError', evalue='Mismatched shell reply ID', traceback=[]))
        
        except queue.Empty:
            if self._verbose:
                print(f"Timeout ({remaining_timeout_for_shell:.1f}s) waiting for shell reply for cell {index}.")
            if not any(out.output_type == 'error' for out in cell_to_run.outputs): # Add error if none present
                cell_to_run.outputs.append(nbformat.v4.new_output(output_type='error', ename='TimeoutError', evalue='Timeout waiting for shell reply', traceback=[]))
        except Exception as e:
            if self._verbose:
                print(f"Error getting or processing shell reply for cell {index}: {e}")
            if not any(out.output_type == 'error' for out in cell_to_run.outputs):
                 cell_to_run.outputs.append(nbformat.v4.new_output(output_type='error', ename='ShellError', evalue=str(e), traceback=[]))

        self._write_notebook_object(nb) 
        return execution_successful
    
    def run_all_cells(self, stop_on_error: bool = True, restart: bool = False) -> int: # return the number of successful cells
        """
        Runs all code cells in the notebook sequentially.
        Returns True if all executed cells were successful, False otherwise.
        """
        if self._verbose:
            print(f"\n--- Running all code cells for session '{self.name}' ---")
        if not self.notebook_file or not U.pexists(self.notebook_file):
            if self._verbose:
                print("Notebook file not found. Cannot run cells.")
            return None
            
        nb = self._read_notebook_object()
        if not nb: return None

        if restart:
            self.start_kernel()
            self.last_stop_index = 0

        failed_cell_idx = None
        for i, cell_data in enumerate(nb.cells):
            if i < self.last_stop_index: continue
            if cell_data.cell_type == 'code':
                if self._verbose:
                    print(f"\nAttempting to run cell {i}...")
                success = self.run_cell(i)
                if not success:
                    failed_cell_idx = i
                    if self._verbose:
                        print(f"!!! Error in cell {i}. Halting execution as stop_on_error is {stop_on_error}.")
                    if stop_on_error:
                        break 
        overall_success = failed_cell_idx is None
        if self._verbose:
            print(f"--- Finished running all cells for session '{self.name}'. Overall success: {overall_success} ---")
        if not overall_success:
            self.last_stop_index = 0 # rerun all when buggy to avoid buggy cells influence the others
        else:
            self.last_stop_index = len(nb.cells) # if no error, update it, no need to rerun the cells
        if restart:
            self.shutdown()
        return failed_cell_idx

    def shutdown(self):
        """Shuts down all resources for this session (server and kernel)."""
        if self._verbose:
            print(f"\n--- Initiating full shutdown for session '{self.name}' ---")
        self.shutdown_server()
        self.shutdown_kernel()
        if self._verbose:
            print(f"--- Full shutdown for session '{self.name}' completed ---")


class JupyterSandbox:
    project_root: str
    _verbose: bool = False

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None, verbose: bool = False):
        self.project_root = config['project_root']
        self.config = config
        self.sandbox_dir = path if path else U.pjoin(U.TMP_DIR, 'sandbox', config['name'])
        if self._verbose:
            print(f"Initializing JupyterSandbox in: {self.sandbox_dir}")
        self.session_dir = U.pjoin(self.sandbox_dir, 'sessions')
        self.active_sessions: Dict[str, JupyterSession] = {} # Track active sessions
        self._verbose = verbose

    def silence(self):
        self._verbose = False
        for sess in self.active_sessions.values():
            sess.silence()

    def verbose(self):
        self._verbose = True
        for sess in self.active_sessions.values():
            sess.verbose()

    def new_session(self, name: Optional[str] = None, metadata: Dict[str, Any] = {}, path = None) -> JupyterSession:
        # project_root is the root directory of the project, it is used to load the proxy 
        proxy_cfg = metadata.get('proxy', {})
        proxy_cfg['activate_proxies'] = proxy_cfg.get('activate_proxies', self.config['activate_proxies'])
        proxy_cfg['cutoff_date'] = proxy_cfg.get('cutoff_date', None)
        proxy_cfg['deploy_mode'] = proxy_cfg.get('deploy_mode', False)
        metadata['project_root'] = self.project_root
        metadata['proxy'] = proxy_cfg

        session_name_base = name if name else dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+uuid.uuid4().hex[:6]
        session_name = session_name_base
        session_path = path if path else U.pjoin(self.session_dir, session_name)
        # count = 0
        # while U.pexists(session_path):
        #     count += 1
        #     session_name = f"{session_name_base}-{count}"
        #     session_path = U.pjoin(self.session_dir, session_name)
        #     if count > 100: raise Exception(f"Could not create unique directory for session '{session_name_base}'")

        if self._verbose:
            print(f"Creating new session '{session_name}' in directory: {session_path}")
        U.mkdirs(session_path)
        notebook_file = U.pjoin(session_path, f"{session_name}.ipynb")
        if U.pexists(notebook_file):
            raise Exception(f"Session '{session_name}' already exists in {session_path}")
        sess = JupyterSession(name=session_name, dir=session_path, metadata=metadata, notebook_file=notebook_file, _verbose=self._verbose)
        sess.init_session()
        sess._ensure_notebook_file(create=True) # Create the .ipynb file immediately
        
        # Save session metadata
        U.save_json(U.pjoin(session_path, f'{session_name}_meta.json'), sess.to_dict())
        self.active_sessions[session_name] = sess # Track it as active
        return sess
    

    def get_session(self, session_name: str, create: bool = True, metadata: Dict[str, Any] = {}, path = None) -> Optional[JupyterSession]: # Changed default create to False
        if session_name in self.active_sessions:
            return self.active_sessions[session_name]
        
        session_path = path if path else U.pjoin(self.session_dir, session_name)
        meta_file = U.pjoin(session_path, f'{session_name}_meta.json')
        
        if U.pexists(meta_file):
            if self._verbose:
                print(f"Loading session '{session_name}' from {meta_file}")
            # try:
            sess_data = U.load_json(meta_file)
            sess = JupyterSession.from_dict(sess_data)
            self.active_sessions[session_name] = sess # Add to active if loaded
            return sess
            # except Exception as e:
            #     print(f"Error loading session '{session_name}': {e}")
            #     return None # Error during load, and not creating new
        elif create:
            if self._verbose:
                print(f"Session '{session_name}' not found. Creating new.")
            return self.new_session(name=session_name, metadata=metadata, path=session_path)
        else:
            if self._verbose:
                print(f"Session '{session_name}' not found and create is False.")
            return None


    def shutdown_session_resources(self, session_name: str):
        if session_name in self.active_sessions:
            session = self.active_sessions[session_name]
            session.shutdown() # This now shuts down server AND kernel
            # del self.active_sessions[session_name] # Keep it in active_sessions, just resources are down
            if self._verbose:
                print(f"Resources for session '{session_name}' shut down.")
        else:
            # Try to load it if not active, to shutdown resources if a meta file exists
            # This is for cases where a session might exist on disk but isn't "active"
            # However, shutdown typically implies an active process.
            if self._verbose:
                print(f"Session '{session_name}' not found in active sessions for resource shutdown.")

    def delete_session_completely(self, session_name: str):
        """Shuts down resources and removes session from disk."""
        if self._verbose:
            print(f"Attempting to completely delete session '{session_name}'...")
        if session_name in self.active_sessions:
            session = self.active_sessions[session_name]
            session.shutdown() # Ensure server/kernel are off
            session_path = session.dir
            del self.active_sessions[session_name]
        else:
            # If not active, construct path from name
            session_path = U.pjoin(self.session_dir, session_name)
            # Attempt to shut down if somehow it was running without being "active"
            # (This part is a bit speculative - usually you'd have an active session to shut down)

        if U.pexists(session_path):
            try:
                U.rmtree(session_path) # Use shutil.rmtree via U
                if self._verbose:
                    print(f"Successfully deleted session directory: {session_path}")
            except Exception as e:
                if self._verbose:
                    print(f"Error deleting session directory {session_path}: {e}")
        else:
            if self._verbose:
                print(f"Session directory {session_path} not found for deletion.")


    def shutdown_all_sessions_resources(self):
        if self._verbose:
            print("Shutting down resources for all active Jupyter sessions...")
        for name in list(self.active_sessions.keys()): # Iterate copy
            self.shutdown_session_resources(name)
        if self._verbose:
            print("Resources for all active sessions shut down.")




