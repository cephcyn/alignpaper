import React from 'react';

import logo from './logo.svg';
import './App.css';

class ShiftButton extends React.Component {
  constructor(props) {
    super(props);
    this.shiftButton = this.shiftButton.bind(this);
  }

  shiftButton(e) {
    e.preventDefault();
    // console.log("Shift button clicked!");
    // console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        row: this.props.rownum,
        col: this.props.colnum,
        shift_dist: this.props.direction,
      })
    };
    fetch("/api/alignop/shift", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering ShiftButton ..............");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    let text;
    if (this.props.direction === -1) {
      text = "<"; // &lt;
    } else {
      text = ">"; // &gt;
    }

    return (
      <button
        className="tight"
        onClick={this.shiftButton}>
          {text}
      </button>)
  }
}

class InsertButton extends React.Component {
  constructor(props) {
    super(props);
    this.insertButton = this.insertButton.bind(this);
  }

  insertButton(e) {
    e.preventDefault();
    console.log("Insert button clicked!");
    console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        col: this.props.colnum,
        insertafter: true,
      })
    };
    fetch("/api/alignop/insertcol", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering InsertButton ...........");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    return (
      <button
        className="tight"
        onClick={this.insertButton}>
          +
      </button>)
  }
}

class DeleteButton extends React.Component {
  constructor(props) {
    super(props);
    this.deleteButton = this.deleteButton.bind(this);
  }

  deleteButton(e) {
    e.preventDefault();
    console.log("Delete button clicked!");
    console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        col: this.props.colnum,
      })
    };
    fetch("/api/alignop/deletecol", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering DeleteButton .............");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    return (
      <button
        className="tight"
        onClick={this.deleteButton}>
          -
      </button>)
  }
}

class AlignmentTable extends React.Component {
  render() {
    // console.log("rerendering AlignmentTable ..............");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    const output = this.props.dataLockCols.map(
      (locked, index) => {
        return (
          <th key={index}>
            txt{index}
            <br/>
            lock
            <input
              key={"collock"+index}
              name={index}
              type="checkbox"
              checked={locked}
              onChange={this.props.handleColLockChange}
            />
          </th>
        );
      }
    );
    const header = (
      <tr key='header'>
        <td></td>
        {output}
      </tr>
    );

    const rows = this.props.data.map(
      (row) => {
        const cols = row.txt.map((cell, index) => {
          return (
            <td key={index}>
              <strong>{cell.join(' ')}</strong>
              <br/>
              <ShiftButton
                data={this.props.data}
                rownum={row.id}
                colnum={index}
                direction={-1}
                onAlignmentChange={this.props.onAlignmentChange}
              />
              <ShiftButton
                data={this.props.data}
                rownum={row.id}
                colnum={index}
                direction={1}
                onAlignmentChange={this.props.onAlignmentChange}
              />
              <br/>
              <InsertButton
                data={this.props.data}
                colnum={index}
                onAlignmentChange={this.props.onAlignmentChange}
              />
              <DeleteButton
                data={this.props.data}
                colnum={index}
                onAlignmentChange={this.props.onAlignmentChange}
              />
            </td>
          );
        });
        return (
          <tr key={row.id}>
            <td className="align-id">{row.id}</td>
            {cols}
          </tr>
        );
      }
    );

    return (
      <table>
        <thead>
          {header}
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    );
  }
}

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      alignment: [],
      alignment_cols_locked: [],
      parse_constituency: {},
      inputvalue: "",
      loading: false,
      textstatus: "",
    };
    this.handleTextChange = this.handleTextChange.bind(this);
    this.handleAlignmentChange = this.handleAlignmentChange.bind(this);
    this.handleColLockChange = this.handleColLockChange.bind(this);
    this.alignRawText = this.alignRawText.bind(this);
    this.updateAlignmentProgress = this.updateAlignmentProgress.bind(this);
    this.alignmentScore = this.alignmentScore.bind(this);
    this.alignmentSearch = this.alignmentSearch.bind(this);
    this.updateSearchProgress = this.updateSearchProgress.bind(this);
    this.buttonDoesNothing = this.buttonDoesNothing.bind(this);
    this.alignDataSave = this.alignDataSave.bind(this);
    this.alignDataLoadClick = this.alignDataLoadClick.bind(this);
    this.alignDataLoad = this.alignDataLoad.bind(this);
  }

  handleTextChange(e) {
    this.setState({ inputvalue: e.target.value });
  }

  handleAlignmentChange(e) {
    // console.log('in handleAlignmentChange');
    this.setState({ alignment: e.alignment });
    // TODO preserve previous cols_locked state somehow?
    if (e.alignment.length > 0) {
      this.setState({ alignment_cols_locked: new Array(e.alignment[0]['txt'].length).fill(false) });
    }
  }

  handleColLockChange(e) {
    // console.log('in handleColLockChange');
    this.setState((prevState, props) => {
      // this is an awful hack for deep cloning this list
      let updated = JSON.parse(JSON.stringify(prevState.alignment_cols_locked));
      updated[e.target.name] = !updated[e.target.name];
      return { alignment_cols_locked: updated };
    });
  }

  alignRawText(e) {
    e.preventDefault();
    console.log("Raw text align button clicked!");
    this.setState({ loading: true });
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        input: this.state.inputvalue
      })
    };
    fetch("/api/textalign", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.updateAlignmentProgress(data['location']);
      });
  }

  updateAlignmentProgress(status_url) {
    fetch(status_url)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        console.log('got update from server...');
        console.log(data);
        if (data['state'] !== 'PENDING' && data['state'] !== 'PROGRESS') {
          if ('alignment' in data) {
            // success!
            this.setState({ alignment: data['alignment'] });
            this.setState({ parse_constituency: data['parse_constituency'] });
            this.setState({ loading: false });
            this.setState({ textstatus: "" });
          } else {
            // failure?
            this.setState({ alignment: [] });
            this.setState({ parse_constituency: {} });
            this.setState({ loading: false });
            this.setState({ textstatus: data['status'] });
          }
        } else {
          // check back on the progress every so often...
          this.setState({ textstatus: data['status'] });
          setTimeout(() => {
            this.updateAlignmentProgress(status_url);
          }, 1000);
        }
      });
  }

  alignmentScore(e) {
    e.preventDefault();
    console.log("alignment score button clicked!");
    this.setState({ loading: true });
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.state.alignment),
      })
    };
    fetch("/api/alignscore", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.setState({ loading: false });
        this.setState(data);
      });
  }

  alignmentSearch(e) {
    e.preventDefault();
    console.log("alignment search button clicked!");
    this.setState({ loading: true });
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.state.alignment),
      })
    };
    fetch("/api/alignsearch", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.updateSearchProgress(data['location']);
      });
  }

  updateSearchProgress(status_url) {
    fetch(status_url)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        console.log('got update from server...');
        console.log(data);
        if (data['state'] !== 'PENDING' && data['state'] !== 'PROGRESS') {
          if ('alignment' in data) {
            // success!
            console.log('setting data.alignment');
            this.setState({ alignment: data['alignment'] });
            console.log('setting data.loading');
            this.setState({ loading: false });
            console.log('setting data.status');
            this.setState({ textstatus: data['status'] });
          } else {
            // failure?
            this.setState({ alignment: [] });
            this.setState({ loading: false });
            this.setState({ textstatus: data['status'] });
          }
        } else {
          // check back on the progress every so often...
          this.setState({ textstatus: data['status'] });
          setTimeout(() => {
            this.updateSearchProgress(status_url);
          }, 1000);
        }
      });
  }

  buttonDoesNothing(e) {
    e.preventDefault();
    console.log("nothing button clicked!");
  }

  alignDataSave(e) {
    e.preventDefault();
    console.log("save button clicked!");
    const output = JSON.stringify({
      alignment: this.state.alignment,
      alignment_cols_locked: this.state.alignment_cols_locked,
      parse_constituency: this.state.parse_constituency,
    });
    const blob = new Blob([output]);
    const fileDownloadUrl = URL.createObjectURL(blob);
    this.setState ({fileDownloadUrl: fileDownloadUrl},
      () => {
        this.dofileDownload.click();
        URL.revokeObjectURL(fileDownloadUrl);  // free up storage--no longer needed.
        this.setState({fileDownloadUrl: ""})
      }
    );
  }

  alignDataLoadClick(e) {
    e.preventDefault();
    console.log("load button clicked!");
    this.dofileUpload.click()
  }

  alignDataLoad(e) {
    const fileObj = e.target.files[0];
    const reader = new FileReader();
    let fileloaded = e => {
      // e.target.result is the file's content as text
      const fileContents = e.target.result;
      this.setState(JSON.parse(fileContents));
    }

    fileloaded = fileloaded.bind(this);
    reader.onload = fileloaded;
    reader.readAsText(fileObj);
  }

  render() {
    console.log("rerendering App.......", new Date());
    console.log("state:", this.state);

    // only render loading indicator spinner if we are currently waiting on the api
    let loadingspinner;
    if (this.state.loading) {
      loadingspinner = <p>Working...</p>
    } else {
      loadingspinner = <br />
    }

    // only render alignment if there's content
    let aligntable;
    if (this.state.alignment.length > 0) {
      aligntable = <AlignmentTable
        data={this.state.alignment}
        dataLockCols={this.state.alignment_cols_locked}
        onAlignmentChange={this.handleAlignmentChange}
        handleColLockChange={this.handleColLockChange}
      />
    } else {
      aligntable = <br />
    }

    return (
      <div className="App">
        <textarea
          value={this.state.inputvalue}
          onChange={this.handleTextChange}
          className="raw-input"
        />
        <br />
        <button onClick={this.alignRawText}>Align Texts</button>
        <button onClick={this.alignmentScore}>Alignment Score</button>
        <button onClick={this.alignmentSearch}>Alignment Search</button>
        <button onClick={this.buttonDoesNothing}>This Button Does Nothing</button>
        <br />
        <button onClick={this.alignDataSave}>Save Alignment</button>
        <a className="hidden"
           download="alignment.json"
           href={this.state.fileDownloadUrl}
           ref={e=>this.dofileDownload = e}
        >download-href</a>
        <button onClick={this.alignDataLoadClick}>Load Alignment</button>
        <input type="file" className="hidden"
            multiple={false}
            accept=".json,.text,application/json"
            onChange={e => this.alignDataLoad(e)}
            ref={e=>this.dofileUpload = e}
          />
        <hr />
        {aligntable}
        {loadingspinner}
        <p>{this.state.textstatus}</p>
        <hr />
        <p>alignment_score is...</p>
        <p>{this.state.alignment_score ? this.state.alignment_score.toString() : 'Undefined'}</p>
        <hr />
        <img src={logo} className="App-logo" alt="logo" />
      </div>
    );
  }
}

export default App;
