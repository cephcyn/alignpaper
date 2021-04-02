import React from 'react';

import logo from './logo.svg';
import './App.css';

class AlignmentTable extends React.Component {
  render() {
    // console.log("rerendering AlignmentTable =========");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    const rows = this.props.data.map(
      (row) => {
        const cols = row.txt.map((cell, index) => {
          return <td key={index}>{cell.join(' ')}</td>
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
      inputvalue: ""
    };

    this.handleChange = this.handleChange.bind(this);
    this.activateLasers = this.activateLasers.bind(this);
  }

  componentDidMount() {}

  handleChange(e) {
    this.setState({inputvalue: e.target.value});
  }

  activateLasers(e) {
    e.preventDefault();
    console.log(e, "Button clicked! value="+this.state.inputvalue);
    fetch(this.props.apiUrl+new URLSearchParams({
      id: "3",
      word: this.state.inputvalue,
    }))
      .then((response) => response.json())
      .then((data) => this.setState(data));
  }

  render() {
    console.log("rerendering App =========");
    console.log("props:", this.props);
    console.log("state:", this.state);

    // only render alignment if there's content
    let aligntable;
    if (this.state.alignment.length > 0) {
      aligntable = <AlignmentTable data={this.state.alignment} />
    } else {
      aligntable = <br />
    }

    return (
      <div className="App">
        <textarea value={this.state.inputvalue} onChange={this.handleChange} />
        <br />
        <button onClick={this.activateLasers}>Submit Request</button>
        <br />
        <br />
        {aligntable}
        <p>temp_arg_data is...</p>
        <p>{this.state.temp_arg_data}</p>
        <br />
        <img src={logo} className="App-logo" alt="logo" />
      </div>
    );
  }
}

export default App;
